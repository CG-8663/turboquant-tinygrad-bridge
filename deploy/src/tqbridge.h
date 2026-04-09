/**
 * tqbridge.h — TurboQuant KV cache compression bridge
 *
 * Standalone C API for TurboQuant PolarQuant compression/decompression.
 * Compress KV cache tensors for cross-device transfer between Metal and CUDA.
 *
 * Usage:
 *   tq_bridge *bridge = NULL;
 *   tq_bridge_init(&bridge, 128, TQ_FORMAT_TURBO3, 42);
 *   tq_compress(bridge, input, n_vectors, &output);
 *   tq_decompress(bridge, &output, result);
 *   tq_bridge_free(bridge);
 */

#ifndef TQBRIDGE_H
#define TQBRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Status codes ─────────────────────────────────────────────── */

typedef enum {
    TQ_STATUS_OK = 0,
    TQ_STATUS_ERROR = -1,
    TQ_STATUS_INVALID_ARG = -2,
    TQ_STATUS_ALLOC_FAILED = -3,
    TQ_STATUS_FORMAT_ERROR = -4,
    TQ_STATUS_DEVICE_ERROR = -5,
    TQ_STATUS_CRC_MISMATCH = -6,
} tq_status;

const char *tq_status_str(tq_status status);

/* ── Compression formats ──────────────────────────────────────── */

typedef enum {
    TQ_FORMAT_FP16   = 0x10,
    TQ_FORMAT_Q8_0   = 0x08,
    TQ_FORMAT_Q5_K_M = 0x05,
    TQ_FORMAT_TURBO4 = 0x04,
    TQ_FORMAT_TURBO3 = 0x03,
    TQ_FORMAT_TURBO2 = 0x02,
} tq_format;

/* ── Wire protocol header (40 bytes, little-endian) ───────────── */

#define TQ_MAGIC       0x54514B56  /* "TQKV" */
#define TQ_HEADER_SIZE 40
#define TQ_HEADER_VER  1

typedef struct {
    tq_format fmt_k;
    tq_format fmt_v;
    uint16_t  n_layers;
    uint16_t  layer_start;
    uint32_t  seq_len;
    uint16_t  n_heads_k;
    uint16_t  n_heads_v;
    uint16_t  head_dim;
    uint16_t  flags;
    uint64_t  payload_bytes;
    uint8_t   version;
} tq_wire_header;

#define TQ_FLAG_ASYMMETRIC_KV (1 << 0)

tq_status tq_encode_header(const tq_wire_header *hdr, uint8_t out[TQ_HEADER_SIZE]);
tq_status tq_decode_header(const uint8_t data[TQ_HEADER_SIZE], tq_wire_header *hdr);

/* ── Compressed payload ───────────────────────────────────────── */

typedef struct {
    void    *data;       /* packed compressed bytes */
    size_t   size;       /* total bytes */
    size_t   n_vectors;  /* number of vectors compressed */
    uint16_t head_dim;   /* dimension per head */
    tq_format fmt;       /* compression format used */
} tq_compressed;

/* ── Bridge context ───────────────────────────────────────────── */

typedef struct tq_bridge tq_bridge;

/**
 * Initialize a bridge context.
 *   head_dim: dimension per attention head (typically 128)
 *   fmt: default compression format
 *   seed: rotation matrix seed (must match on both endpoints, default 42)
 */
tq_status tq_bridge_init(tq_bridge **bridge, int head_dim, tq_format fmt, int seed);

/**
 * Initialize with precomputed rotation matrix and codebook from Python.
 * Ensures bit-exact parity with the NumPy reference implementation.
 *   rotation: (head_dim * head_dim) float32 values, row-major
 *   codebook: (2^bit_width) float32 values, sorted ascending
 *   n_centroids: number of codebook entries (must equal 2^bit_width)
 */
tq_status tq_bridge_init_precomputed(tq_bridge **bridge, int head_dim, tq_format fmt,
                                     const float *rotation, const float *codebook,
                                     int n_centroids);

void      tq_bridge_free(tq_bridge *bridge);

/* ── Core compression ops ─────────────────────────────────────── */

/**
 * Compress float32 vectors using PolarQuant.
 *   input: (n_vectors * head_dim) float32 values
 *   n_vectors: number of vectors to compress
 *   output: caller-provided, will be filled with compressed data
 */
tq_status tq_compress(tq_bridge *bridge,
                      const float *input, size_t n_vectors,
                      tq_compressed *output);

/**
 * Decompress back to float32.
 *   input: compressed payload from tq_compress
 *   output: (n_vectors * head_dim) float32 values, caller-allocated
 */
tq_status tq_decompress(tq_bridge *bridge,
                        const tq_compressed *input,
                        float *output);

/**
 * Free compressed payload data.
 */
void tq_compressed_free(tq_compressed *c);

/* ── Q8_0 block format (ggml-compatible) ──────────────────────── */

#define TQ_QK8_0 32  /* block size */

tq_status tq_compress_q8_0(const float *input, size_t n_elements,
                           void **out_data, size_t *out_size);
tq_status tq_decompress_q8_0(const void *data, size_t n_elements,
                             float *output);

/* ── Utility ──────────────────────────────────────────────────── */

/** Compression ratio for a given format and head_dim. */
float tq_compression_ratio(tq_format fmt, int head_dim);

/** Compressed size in bytes for n_vectors at given format. */
size_t tq_compressed_size(tq_format fmt, int head_dim, size_t n_vectors);

#ifdef __cplusplus
}
#endif

#endif /* TQBRIDGE_H */
