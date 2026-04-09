/**
 * tqbridge-server — Decode node server (replaces Python serve_decode.py).
 *
 * Listens for compressed KV cache over TCP, decompresses via C library,
 * prints stats. Deploy on GX10 nodes or any machine.
 *
 * Usage:
 *   tqbridge-server --port 9473 --head-dim 128 --seed 42
 */

#include "tqbridge.h"
#include "tqbridge_net.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>

static tq_bridge *g_bridge_turbo3 = NULL;
static tq_bridge *g_bridge_turbo4 = NULL;
static int g_tokens = 0;
static double g_start_time = 0.0;

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void on_kv_received(const tq_wire_header *header,
                            const uint8_t *k_data, size_t k_size,
                            const uint8_t *v_data, size_t v_size,
                            void *user_data) {
    (void)user_data;
    double t0 = now_sec();

    int head_dim = header->head_dim;
    int n_layers = header->n_layers;

    /* Select bridge based on format */
    tq_bridge *bridge = NULL;
    if (header->fmt_v == TQ_FORMAT_TURBO3) bridge = g_bridge_turbo3;
    else if (header->fmt_v == TQ_FORMAT_TURBO4) bridge = g_bridge_turbo4;

    if (bridge) {
        /* Decompress K */
        tq_compressed k_comp = {
            .data = (void *)k_data, .size = k_size,
            .n_vectors = (size_t)(header->n_heads_k * header->seq_len * n_layers),
            .head_dim = (uint16_t)head_dim,
            .fmt = header->fmt_k,
        };

        size_t k_elements = k_comp.n_vectors * head_dim;
        float *k_out = (float *)malloc(k_elements * sizeof(float));
        if (k_out) {
            tq_decompress(bridge, &k_comp, k_out);
            free(k_out);
        }

        /* Decompress V */
        tq_compressed v_comp = {
            .data = (void *)v_data, .size = v_size,
            .n_vectors = (size_t)(header->n_heads_v * header->seq_len * n_layers),
            .head_dim = (uint16_t)head_dim,
            .fmt = header->fmt_v,
        };

        size_t v_elements = v_comp.n_vectors * head_dim;
        float *v_out = (float *)malloc(v_elements * sizeof(float));
        if (v_out) {
            tq_decompress(bridge, &v_comp, v_out);
            free(v_out);
        }
    }

    double t1 = now_sec();
    double decompress_ms = (t1 - t0) * 1000.0;

    g_tokens++;
    double elapsed = t1 - g_start_time;
    double tps = elapsed > 0 ? g_tokens / elapsed : 0;

    printf("  [%4d] layers %d-%d  decompress %.1fms  %zu+%zu bytes  (%.1f tok/s avg)\n",
           g_tokens, header->layer_start,
           header->layer_start + n_layers - 1,
           decompress_ms, k_size, v_size, tps);
}

static volatile int g_running = 1;

static void signal_handler(int sig) {
    (void)sig;
    g_running = 0;
}

int main(int argc, char **argv) {
    int port = 9473;
    int head_dim = 128;
    int seed = 42;

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) port = atoi(argv[++i]);
        else if (strcmp(argv[i], "--head-dim") == 0 && i + 1 < argc) head_dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: tqbridge-server [--port PORT] [--head-dim DIM] [--seed SEED]\n");
            return 0;
        }
    }

    /* Init bridges for turbo3 and turbo4 */
    tq_status st;
    st = tq_bridge_init(&g_bridge_turbo3, head_dim, TQ_FORMAT_TURBO3, seed);
    if (st != TQ_STATUS_OK) {
        fprintf(stderr, "Failed to init turbo3 bridge: %s\n", tq_status_str(st));
        return 1;
    }
    st = tq_bridge_init(&g_bridge_turbo4, head_dim, TQ_FORMAT_TURBO4, seed);
    if (st != TQ_STATUS_OK) {
        fprintf(stderr, "Failed to init turbo4 bridge: %s\n", tq_status_str(st));
        return 1;
    }

    printf("[tqbridge-server] Listening on port %d (head_dim=%d, seed=%d)\n", port, head_dim, seed);
    printf("[tqbridge-server] Ctrl+C to stop\n\n");

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    g_start_time = now_sec();

    tq_receiver receiver;
    tq_receiver_init(&receiver, port);
    tq_receiver_start(&receiver, on_kv_received, NULL);

    /* Cleanup */
    printf("\n[tqbridge-server] Stopped. %d tokens received.\n", g_tokens);
    tq_bridge_free(g_bridge_turbo3);
    tq_bridge_free(g_bridge_turbo4);
    return 0;
}
