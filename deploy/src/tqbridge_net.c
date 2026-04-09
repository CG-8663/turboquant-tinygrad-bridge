/**
 * tqbridge_net.c — TCP transport for TurboQuant KV cache bridge.
 *
 * Pure C, no dependencies beyond POSIX sockets and tqbridge.h.
 */

#include "tqbridge_net.h"
#include "tqbridge.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <poll.h>

/* ── TCP Sender ──────────────────────────────────────────────── */

tq_status tq_sender_init(tq_sender *s, const char *host, int port,
                          float timeout_s, int max_retries) {
    if (!s || !host) return TQ_STATUS_INVALID_ARG;
    memset(s, 0, sizeof(*s));
    strncpy(s->host, host, sizeof(s->host) - 1);
    s->port = port;
    s->timeout_s = timeout_s > 0 ? timeout_s : 10.0f;
    s->max_retries = max_retries > 0 ? max_retries : 3;
    s->fd = -1;
    s->connected = 0;
    return TQ_STATUS_OK;
}

tq_status tq_sender_connect(tq_sender *s) {
    if (!s) return TQ_STATUS_INVALID_ARG;

    /* Close existing connection */
    tq_sender_close(s);

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return TQ_STATUS_DEVICE_ERROR;

    /* Set TCP_NODELAY */
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    /* Set send buffer */
    int bufsize = 1 << 20;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &bufsize, sizeof(bufsize));

    /* Set timeout */
    struct timeval tv;
    tv.tv_sec = (int)s->timeout_s;
    tv.tv_usec = (int)((s->timeout_s - tv.tv_sec) * 1e6);
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(s->port);

    if (inet_pton(AF_INET, s->host, &addr.sin_addr) != 1) {
        close(fd);
        return TQ_STATUS_INVALID_ARG;
    }

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd);
        return TQ_STATUS_DEVICE_ERROR;
    }

    s->fd = fd;
    s->connected = 1;
    return TQ_STATUS_OK;
}

static tq_status send_all(int fd, const void *buf, size_t len) {
    const uint8_t *p = (const uint8_t *)buf;
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(fd, p + sent, len - sent, 0);
        if (n <= 0) {
            if (errno == EINTR) continue;
            return TQ_STATUS_DEVICE_ERROR;
        }
        sent += (size_t)n;
    }
    return TQ_STATUS_OK;
}

tq_status tq_sender_send_kv(tq_sender *s,
                             const uint8_t *k_data, size_t k_size,
                             const uint8_t *v_data, size_t v_size,
                             const tq_wire_header *header) {
    if (!s || !k_data || !v_data || !header)
        return TQ_STATUS_INVALID_ARG;

    /* Encode wire header */
    uint8_t hdr_buf[TQ_HEADER_SIZE];
    tq_wire_header hdr_copy = *header;
    hdr_copy.payload_bytes = k_size + v_size;
    tq_status st = tq_encode_header(&hdr_copy, hdr_buf);
    if (st != TQ_STATUS_OK) return st;

    /* Retry loop */
    for (int attempt = 0; attempt < s->max_retries; attempt++) {
        if (!s->connected) {
            st = tq_sender_connect(s);
            if (st != TQ_STATUS_OK) {
                if (attempt < s->max_retries - 1) {
                    usleep(100000 * (attempt + 1)); /* backoff */
                    continue;
                }
                return st;
            }
        }

        /* Send: header + K data + V data */
        st = send_all(s->fd, hdr_buf, TQ_HEADER_SIZE);
        if (st == TQ_STATUS_OK) st = send_all(s->fd, k_data, k_size);
        if (st == TQ_STATUS_OK) st = send_all(s->fd, v_data, v_size);

        if (st == TQ_STATUS_OK) return TQ_STATUS_OK;

        /* Send failed — close and retry */
        tq_sender_close(s);
        if (attempt < s->max_retries - 1) {
            usleep(100000 * (attempt + 1));
        }
    }

    return TQ_STATUS_DEVICE_ERROR;
}

void tq_sender_close(tq_sender *s) {
    if (s && s->fd >= 0) {
        close(s->fd);
        s->fd = -1;
        s->connected = 0;
    }
}

/* ── TCP Receiver ────────────────────────────────────────────── */

static int recv_exact(int fd, void *buf, size_t len) {
    uint8_t *p = (uint8_t *)buf;
    size_t got = 0;
    while (got < len) {
        ssize_t n = recv(fd, p + got, len - got, 0);
        if (n <= 0) {
            if (n == 0) return -1; /* connection closed */
            if (errno == EINTR) continue;
            return -1;
        }
        got += (size_t)n;
    }
    return 0;
}

tq_status tq_receiver_init(tq_receiver *r, int port) {
    if (!r) return TQ_STATUS_INVALID_ARG;
    memset(r, 0, sizeof(*r));
    r->port = port;
    r->fd = -1;
    return TQ_STATUS_OK;
}

tq_status tq_receiver_start(tq_receiver *r, tq_on_receive_fn callback, void *user_data) {
    if (!r || !callback) return TQ_STATUS_INVALID_ARG;

    r->on_receive = callback;
    r->user_data = user_data;

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return TQ_STATUS_DEVICE_ERROR;

    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    int bufsize = 1 << 20;
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &bufsize, sizeof(bufsize));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(r->port);

    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd);
        return TQ_STATUS_DEVICE_ERROR;
    }

    if (listen(fd, 4) < 0) {
        close(fd);
        return TQ_STATUS_DEVICE_ERROR;
    }

    r->fd = fd;
    r->running = 1;

    /* Accept loop — runs in the calling thread (use pthread for background) */
    while (r->running) {
        struct pollfd pfd = {.fd = fd, .events = POLLIN};
        int ret = poll(&pfd, 1, 1000); /* 1s timeout */
        if (ret <= 0) continue;

        int client = accept(fd, NULL, NULL);
        if (client < 0) continue;

        /* Handle client */
        while (r->running) {
            uint8_t hdr_buf[TQ_HEADER_SIZE];
            if (recv_exact(client, hdr_buf, TQ_HEADER_SIZE) < 0) break;

            tq_wire_header header;
            tq_status st = tq_decode_header(hdr_buf, &header);
            if (st != TQ_STATUS_OK) break;

            /* Read payload */
            uint8_t *payload = (uint8_t *)malloc(header.payload_bytes);
            if (!payload) break;
            if (recv_exact(client, payload, header.payload_bytes) < 0) {
                free(payload);
                break;
            }

            /* Split K and V (each half) */
            size_t mid = header.payload_bytes / 2;
            r->on_receive(&header, payload, mid, payload + mid,
                          header.payload_bytes - mid, r->user_data);
            free(payload);
        }

        close(client);
    }

    return TQ_STATUS_OK;
}

void tq_receiver_stop(tq_receiver *r) {
    if (r) {
        r->running = 0;
        if (r->fd >= 0) {
            close(r->fd);
            r->fd = -1;
        }
    }
}

/* ── Convenience: compress + send ────────────────────────────── */

tq_status tq_send_compressed(tq_sender *s, tq_bridge *bridge,
                              const float *k_input, const float *v_input,
                              size_t n_vectors, tq_format fmt_k, tq_format fmt_v,
                              int n_layers, int layer_start,
                              int n_heads_k, int n_heads_v, int head_dim, int seq_len) {
    if (!s || !bridge || !k_input || !v_input)
        return TQ_STATUS_INVALID_ARG;

    /* Compress K */
    tq_compressed k_comp = {0}, v_comp = {0};
    tq_status st = tq_compress(bridge, k_input, n_vectors, &k_comp);
    if (st != TQ_STATUS_OK) return st;

    st = tq_compress(bridge, v_input, n_vectors, &v_comp);
    if (st != TQ_STATUS_OK) {
        tq_compressed_free(&k_comp);
        return st;
    }

    /* Build header */
    tq_wire_header header = {
        .fmt_k = fmt_k,
        .fmt_v = fmt_v,
        .n_layers = (uint16_t)n_layers,
        .layer_start = (uint16_t)layer_start,
        .seq_len = (uint32_t)seq_len,
        .n_heads_k = (uint16_t)n_heads_k,
        .n_heads_v = (uint16_t)n_heads_v,
        .head_dim = (uint16_t)head_dim,
        .flags = 0,
        .payload_bytes = k_comp.size + v_comp.size,
        .version = TQ_HEADER_VER,
    };

    /* Send */
    st = tq_sender_send_kv(s, (const uint8_t *)k_comp.data, k_comp.size,
                            (const uint8_t *)v_comp.data, v_comp.size, &header);

    tq_compressed_free(&k_comp);
    tq_compressed_free(&v_comp);
    return st;
}
