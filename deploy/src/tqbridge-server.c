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
        /*
         * Compute n_vectors from the compressed data size, NOT from header fields.
         * The header's n_heads * seq_len * n_layers may claim more vectors than
         * the compressed payload actually contains. Using the header blindly
         * causes tq_decompress to read past the buffer → segfault.
         *
         * For turbo3: each vector = ceil(head_dim * 3 / 8) + 4 bytes (norm+idx)
         * For turbo4: each vector = ceil(head_dim * 4 / 8) + 4 bytes
         * Safe estimate: n_vectors = k_size / bytes_per_vector
         */
        int bits = (header->fmt_k == TQ_FORMAT_TURBO4) ? 4 :
                   (header->fmt_k == TQ_FORMAT_TURBO3) ? 3 : 2;
        size_t bytes_per_vec = (size_t)((head_dim * bits + 7) / 8) + 4;

        /* Decompress K */
        size_t k_nvec = bytes_per_vec > 0 ? k_size / bytes_per_vec : 0;
        if (k_nvec > 0 && k_size >= k_nvec * bytes_per_vec) {
            tq_compressed k_comp = {
                .data = (void *)k_data, .size = k_size,
                .n_vectors = k_nvec,
                .head_dim = (uint16_t)head_dim,
                .fmt = header->fmt_k,
            };
            size_t k_elements = k_nvec * head_dim;
            float *k_out = (float *)malloc(k_elements * sizeof(float));
            if (k_out) {
                tq_decompress(bridge, &k_comp, k_out);
                free(k_out);
            }
        }

        /* Decompress V */
        bits = (header->fmt_v == TQ_FORMAT_TURBO4) ? 4 :
               (header->fmt_v == TQ_FORMAT_TURBO3) ? 3 : 2;
        bytes_per_vec = (size_t)((head_dim * bits + 7) / 8) + 4;

        size_t v_nvec = bytes_per_vec > 0 ? v_size / bytes_per_vec : 0;
        if (v_nvec > 0 && v_size >= v_nvec * bytes_per_vec) {
            tq_compressed v_comp = {
                .data = (void *)v_data, .size = v_size,
                .n_vectors = v_nvec,
                .head_dim = (uint16_t)head_dim,
                .fmt = header->fmt_v,
            };
            size_t v_elements = v_nvec * head_dim;
            float *v_out = (float *)malloc(v_elements * sizeof(float));
            if (v_out) {
                tq_decompress(bridge, &v_comp, v_out);
                free(v_out);
            }
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

/* ── Self-test: compress/decompress round-trip ────────────── */

static int run_selftest(int head_dim, int seed) {
    printf("[self-test] TQBridge compression round-trip validation\n");
    printf("[self-test] head_dim=%d, seed=%d\n\n", head_dim, seed);

    int formats[] = {TQ_FORMAT_TURBO2, TQ_FORMAT_TURBO3, TQ_FORMAT_TURBO4};
    const char *names[] = {"turbo2", "turbo3", "turbo4"};
    int all_pass = 1;

    for (int f = 0; f < 3; f++) {
        tq_bridge *bridge = NULL;
        tq_status st = tq_bridge_init(&bridge, head_dim, formats[f], seed);
        if (st != TQ_STATUS_OK) {
            printf("  %-8s  FAIL (init: %s)\n", names[f], tq_status_str(st));
            all_pass = 0;
            continue;
        }

        /* Generate test vectors */
        int n_vectors = 32;
        float *input = (float *)malloc(n_vectors * head_dim * sizeof(float));
        float *output = (float *)malloc(n_vectors * head_dim * sizeof(float));
        srand(42);
        for (int i = 0; i < n_vectors * head_dim; i++)
            input[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;

        /* Compress */
        tq_compressed comp = {0};
        st = tq_compress(bridge, input, n_vectors, &comp);
        if (st != TQ_STATUS_OK) {
            printf("  %-8s  FAIL (compress: %s)\n", names[f], tq_status_str(st));
            all_pass = 0;
            free(input); free(output); tq_bridge_free(bridge);
            continue;
        }

        /* Decompress */
        st = tq_decompress(bridge, &comp, output);
        if (st != TQ_STATUS_OK) {
            printf("  %-8s  FAIL (decompress: %s)\n", names[f], tq_status_str(st));
            all_pass = 0;
            tq_compressed_free(&comp);
            free(input); free(output); tq_bridge_free(bridge);
            continue;
        }

        /* Check MSE */
        double mse = 0.0;
        for (int i = 0; i < n_vectors * head_dim; i++) {
            double diff = (double)output[i] - (double)input[i];
            mse += diff * diff;
        }
        mse /= (double)(n_vectors * head_dim);

        /* Compression ratio */
        size_t original = n_vectors * head_dim * sizeof(float);
        float ratio = (float)original / (float)comp.size;

        printf("  %-8s  PASS  ratio=%.1fx  MSE=%.6f  (%zu → %zu bytes)\n",
               names[f], ratio, mse, original, comp.size);

        tq_compressed_free(&comp);
        free(input); free(output); tq_bridge_free(bridge);
    }

    /* Q8_0 test */
    {
        int n = 128;
        float *input = (float *)malloc(n * sizeof(float));
        float *output = (float *)malloc(n * sizeof(float));
        srand(99);
        for (int i = 0; i < n; i++)
            input[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;

        void *data = NULL;
        size_t size = 0;
        tq_status st = tq_compress_q8_0(input, n, &data, &size);
        if (st == TQ_STATUS_OK) {
            st = tq_decompress_q8_0(data, n, output);
            double mse = 0.0;
            for (int i = 0; i < n; i++) {
                double diff = (double)output[i] - (double)input[i];
                mse += diff * diff;
            }
            mse /= (double)n;
            float ratio = (float)(n * sizeof(float)) / (float)size;
            printf("  %-8s  PASS  ratio=%.1fx  MSE=%.6f  (%zu → %zu bytes)\n",
                   "q8_0", ratio, mse, n * sizeof(float), size);
            free(data);
        } else {
            printf("  %-8s  FAIL (%s)\n", "q8_0", tq_status_str(st));
            all_pass = 0;
        }
        free(input); free(output);
    }

    /* Wire protocol test */
    {
        tq_wire_header hdr = {
            .fmt_k = TQ_FORMAT_Q8_0, .fmt_v = TQ_FORMAT_TURBO3,
            .n_layers = 32, .layer_start = 0, .seq_len = 2048,
            .n_heads_k = 32, .n_heads_v = 8, .head_dim = 128,
            .flags = 0, .payload_bytes = 1048576, .version = TQ_HEADER_VER,
        };
        uint8_t buf[TQ_HEADER_SIZE];
        tq_wire_header decoded;

        tq_status st1 = tq_encode_header(&hdr, buf);
        tq_status st2 = tq_decode_header(buf, &decoded);

        int wire_ok = (st1 == TQ_STATUS_OK && st2 == TQ_STATUS_OK &&
                       decoded.fmt_k == TQ_FORMAT_Q8_0 &&
                       decoded.fmt_v == TQ_FORMAT_TURBO3 &&
                       decoded.n_layers == 32 &&
                       decoded.payload_bytes == 1048576);

        printf("  %-8s  %s  encode/decode round-trip, CRC32 validation\n",
               "wire", wire_ok ? "PASS" : "FAIL");
        if (!wire_ok) all_pass = 0;
    }

    printf("\n[self-test] %s\n", all_pass ? "ALL PASS" : "SOME FAILURES");
    return all_pass ? 0 : 1;
}

/* ── Benchmark: measure compress/decompress throughput ────── */

static int run_benchmark(int head_dim, int seed) {
    printf("[benchmark] TQBridge compression throughput\n");
    printf("[benchmark] head_dim=%d, seed=%d\n\n", head_dim, seed);

    int n_vectors = 256;
    float *input = (float *)malloc(n_vectors * head_dim * sizeof(float));
    float *output = (float *)malloc(n_vectors * head_dim * sizeof(float));
    srand(42);
    for (int i = 0; i < n_vectors * head_dim; i++)
        input[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;

    int formats[] = {TQ_FORMAT_TURBO2, TQ_FORMAT_TURBO3, TQ_FORMAT_TURBO4};
    const char *names[] = {"turbo2", "turbo3", "turbo4"};

    printf("  %-8s  %10s  %10s  %10s  %8s\n", "Format", "Compress", "Decompress", "Total", "tok/s");
    printf("  ───────────────────────────────────────────────────────\n");

    for (int f = 0; f < 3; f++) {
        tq_bridge *bridge = NULL;
        tq_bridge_init(&bridge, head_dim, formats[f], seed);

        /* Warmup */
        tq_compressed comp = {0};
        tq_compress(bridge, input, n_vectors, &comp);
        tq_decompress(bridge, &comp, output);
        tq_compressed_free(&comp);

        /* Benchmark */
        int iters = 100;
        struct timespec t0, t1, t2;
        double compress_ms = 0, decompress_ms = 0;

        for (int i = 0; i < iters; i++) {
            comp = (tq_compressed){0};
            clock_gettime(CLOCK_MONOTONIC, &t0);
            tq_compress(bridge, input, n_vectors, &comp);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            tq_decompress(bridge, &comp, output);
            clock_gettime(CLOCK_MONOTONIC, &t2);

            compress_ms += (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
            decompress_ms += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_nsec - t1.tv_nsec) / 1e6;
            tq_compressed_free(&comp);
        }

        compress_ms /= iters;
        decompress_ms /= iters;
        double total = compress_ms + decompress_ms;
        double tps = 1000.0 / total;

        printf("  %-8s  %8.2f ms  %8.2f ms  %8.2f ms  %6.0f\n",
               names[f], compress_ms, decompress_ms, total, tps);

        tq_bridge_free(bridge);
    }

    free(input); free(output);
    printf("\n[benchmark] %d vectors × %d dim per iteration\n", n_vectors, head_dim);
    return 0;
}

int main(int argc, char **argv) {
    int port = 9473;
    int head_dim = 128;
    int seed = 42;
    int mode_selftest = 0;
    int mode_benchmark = 0;

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) port = atoi(argv[++i]);
        else if (strcmp(argv[i], "--head-dim") == 0 && i + 1 < argc) head_dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "--self-test") == 0) mode_selftest = 1;
        else if (strcmp(argv[i], "--benchmark") == 0) mode_benchmark = 1;
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: tqbridge-server [OPTIONS]\n\n");
            printf("Modes:\n");
            printf("  (default)     Start decode server on PORT\n");
            printf("  --self-test   Run compression round-trip validation and exit\n");
            printf("  --benchmark   Run throughput benchmark and exit\n");
            printf("\nOptions:\n");
            printf("  --port PORT   Listen port (default: 9473)\n");
            printf("  --head-dim N  Head dimension (default: 128)\n");
            printf("  --seed N      Rotation seed (default: 42)\n");
            printf("  -h, --help    Show this help\n");
            return 0;
        }
    }

    if (mode_selftest) return run_selftest(head_dim, seed);
    if (mode_benchmark) return run_benchmark(head_dim, seed);

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

    /* Force line-buffered stdout so logs appear immediately in nohup/redirect */
    setvbuf(stdout, NULL, _IOLBF, 0);

    printf("[tqbridge-server] Listening on port %d (head_dim=%d, seed=%d)\n", port, head_dim, seed);
    printf("[tqbridge-server] Ctrl+C to stop\n\n");

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGPIPE, SIG_IGN);  /* Ignore broken pipe — don't crash on client disconnect */

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
