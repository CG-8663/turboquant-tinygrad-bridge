/**
 * tqbridge_net.h — TCP transport for TurboQuant KV cache bridge.
 *
 * Send and receive compressed KV cache over TCP with wire protocol headers.
 * Handles connection, retries, and framing.
 */

#ifndef TQBRIDGE_NET_H
#define TQBRIDGE_NET_H

#include "tqbridge.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── TCP Sender ──────────────────────────────────────────────── */

typedef struct {
    int       fd;
    char      host[256];
    int       port;
    float     timeout_s;
    int       max_retries;
    int       connected;
} tq_sender;

tq_status tq_sender_init(tq_sender *s, const char *host, int port,
                          float timeout_s, int max_retries);
tq_status tq_sender_connect(tq_sender *s);
tq_status tq_sender_send_kv(tq_sender *s,
                             const uint8_t *k_data, size_t k_size,
                             const uint8_t *v_data, size_t v_size,
                             const tq_wire_header *header);
void      tq_sender_close(tq_sender *s);

/* ── TCP Receiver ────────────────────────────────────────────── */

typedef void (*tq_on_receive_fn)(const tq_wire_header *header,
                                  const uint8_t *k_data, size_t k_size,
                                  const uint8_t *v_data, size_t v_size,
                                  void *user_data);

typedef struct {
    int              fd;
    int              port;
    int              running;
    tq_on_receive_fn on_receive;
    void            *user_data;
} tq_receiver;

tq_status tq_receiver_init(tq_receiver *r, int port);
tq_status tq_receiver_start(tq_receiver *r, tq_on_receive_fn callback, void *user_data);
void      tq_receiver_stop(tq_receiver *r);

/* ── Utility ─────────────────────────────────────────────────── */

/** Compress and send KV vectors to a remote node in one call. */
tq_status tq_send_compressed(tq_sender *s, tq_bridge *bridge,
                              const float *k_input, const float *v_input,
                              size_t n_vectors, tq_format fmt_k, tq_format fmt_v,
                              int n_layers, int layer_start,
                              int n_heads_k, int n_heads_v, int head_dim, int seq_len);

#ifdef __cplusplus
}
#endif

#endif /* TQBRIDGE_NET_H */
