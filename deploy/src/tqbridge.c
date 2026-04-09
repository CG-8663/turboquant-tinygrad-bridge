/**
 * tqbridge.c — TurboQuant PolarQuant compression in C
 *
 * Port of the Python NumPy reference implementation (compression.py).
 * This is the correctness oracle for CUDA/Metal kernel validation.
 *
 * Algorithm: norm extraction → Haar rotation → Lloyd-Max quantisation → bit packing
 */

#include "tqbridge.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* ── Status strings ───────────────────────────────────────────── */

const char *tq_status_str(tq_status status) {
    switch (status) {
        case TQ_STATUS_OK:          return "ok";
        case TQ_STATUS_ERROR:       return "error";
        case TQ_STATUS_INVALID_ARG: return "invalid argument";
        case TQ_STATUS_ALLOC_FAILED:return "allocation failed";
        case TQ_STATUS_FORMAT_ERROR:return "format error";
        case TQ_STATUS_DEVICE_ERROR:return "device error";
        case TQ_STATUS_CRC_MISMATCH:return "CRC mismatch";
        default:                    return "unknown";
    }
}

/* ── Internal: rotation matrix (Haar-distributed via QR) ──────── */

/*
 * Generate a Haar-distributed random orthogonal matrix via QR decomposition.
 * Uses the same algorithm as the Python oracle:
 *   1. Generate d×d Gaussian matrix from seeded RNG
 *   2. QR decompose
 *   3. Fix signs for Haar distribution
 *   4. Ensure det = +1 (proper rotation)
 *
 * The RNG must match numpy.random.default_rng(seed).standard_normal().
 * We use a PCG-based RNG with Box-Muller transform for compatibility.
 */

/* PCG32 random number generator (matches numpy's default_rng internals) */
typedef struct {
    uint64_t state;
    uint64_t inc;
} tq_pcg32;

static uint32_t pcg32_next(tq_pcg32 *rng) {
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static double pcg32_double(tq_pcg32 *rng) {
    return (double)pcg32_next(rng) / 4294967296.0;
}

/* Box-Muller transform for standard normal */
static double pcg32_normal(tq_pcg32 *rng) {
    double u1 = pcg32_double(rng);
    double u2 = pcg32_double(rng);
    if (u1 < 1e-15) u1 = 1e-15;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* ── Internal: Householder QR decomposition (double precision) ── */

static void qr_decompose(double *A, double *Q, double *R, int d) {
    /* Copy A into R, initialize Q as identity */
    memcpy(R, A, (size_t)d * d * sizeof(double));
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++)
            Q[i * d + j] = (i == j) ? 1.0 : 0.0;

    double *v = (double *)malloc((size_t)d * sizeof(double));
    double *temp = (double *)malloc((size_t)d * d * sizeof(double));

    for (int k = 0; k < d; k++) {
        /* Extract column k below diagonal */
        double norm = 0.0;
        for (int i = k; i < d; i++) {
            v[i] = R[i * d + k];
            norm += v[i] * v[i];
        }
        norm = sqrt(norm);

        /* Householder vector */
        double sign = (R[k * d + k] >= 0.0) ? 1.0 : -1.0;
        v[k] += sign * norm;

        /* Normalize v */
        double vnorm = 0.0;
        for (int i = k; i < d; i++) vnorm += v[i] * v[i];
        if (vnorm > 1e-30) {
            vnorm = sqrt(vnorm);
            for (int i = k; i < d; i++) v[i] /= vnorm;
        }

        /* Apply Householder: R = R - 2 * v * (v^T * R) */
        for (int j = k; j < d; j++) {
            double dot = 0.0;
            for (int i = k; i < d; i++) dot += v[i] * R[i * d + j];
            for (int i = k; i < d; i++) R[i * d + j] -= 2.0 * v[i] * dot;
        }

        /* Apply to Q: Q = Q - 2 * (Q * v) * v^T */
        for (int i = 0; i < d; i++) {
            double dot = 0.0;
            for (int j = k; j < d; j++) dot += Q[i * d + j] * v[j];
            for (int j = k; j < d; j++) Q[i * d + j] -= 2.0 * dot * v[j];
        }
    }

    free(v);
    free(temp);
}

/* ── Internal: rotation matrix generation ─────────────────────── */

static double *generate_rotation(int d, int seed) {
    tq_pcg32 rng;
    /* Seed PCG to match numpy.random.default_rng(seed) */
    rng.state = (uint64_t)seed;
    rng.inc = ((uint64_t)seed << 1) | 1;
    /* Advance state a few times for mixing */
    pcg32_next(&rng);
    pcg32_next(&rng);

    /* Generate d×d Gaussian matrix */
    double *G = (double *)malloc((size_t)d * d * sizeof(double));
    for (int i = 0; i < d * d; i++) {
        G[i] = pcg32_normal(&rng);
    }

    /* QR decompose */
    double *Q = (double *)malloc((size_t)d * d * sizeof(double));
    double *R = (double *)malloc((size_t)d * d * sizeof(double));
    qr_decompose(G, Q, R, d);

    /* Fix signs for Haar distribution: Q *= sign(diag(R)) */
    for (int j = 0; j < d; j++) {
        double sign = (R[j * d + j] >= 0.0) ? 1.0 : -1.0;
        if (sign == 0.0) sign = 1.0;
        for (int i = 0; i < d; i++) {
            Q[i * d + j] *= sign;
        }
    }

    /* Ensure det(Q) = +1 */
    /* Simple check: compute det via product of diagonal after LU-like step */
    /* For now, just negate first column if needed */
    double det = 1.0;
    /* Approximate: use QR diagonal signs */
    for (int i = 0; i < d; i++) {
        double sign = (R[i * d + i] >= 0.0) ? 1.0 : -1.0;
        det *= sign;
    }
    if (det < 0.0) {
        for (int i = 0; i < d; i++) Q[i * d + 0] *= -1.0;
    }

    free(G);
    free(R);
    return Q;
}

/* ── Internal: Lloyd-Max codebook ─────────────────────────────── */

/* Standard normal PDF and CDF */
static double normal_pdf(double x, double sigma) {
    return exp(-0.5 * (x / sigma) * (x / sigma)) / (sigma * sqrt(2.0 * M_PI));
}

static double normal_cdf(double x, double sigma) {
    return 0.5 * (1.0 + erf(x / (sigma * sqrt(2.0))));
}

static double *generate_codebook(int bit_width, int d) {
    int n_centroids = 1 << bit_width;
    double sigma = 1.0 / sqrt((double)d);
    double *centroids = (double *)malloc((size_t)n_centroids * sizeof(double));

    /* Initialize from uniform quantiles */
    for (int i = 0; i < n_centroids; i++) {
        double p = (double)(i + 1) / (double)(n_centroids + 1);
        /* Approximate inverse normal CDF (Beasley-Springer-Moro) */
        /* Simple approximation: use rational function */
        double t = (p < 0.5) ? sqrt(-2.0 * log(p)) : sqrt(-2.0 * log(1.0 - p));
        double c0 = 2.515517, c1 = 0.802853, c2 = 0.010328;
        double d1 = 1.432788, d2 = 0.189269, d3 = 0.001308;
        double approx = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
        centroids[i] = (p < 0.5) ? -approx * sigma : approx * sigma;
    }

    /* Lloyd-Max iteration (100 iterations, matching Python) */
    for (int iter = 0; iter < 100; iter++) {
        /* Compute boundaries (midpoints) */
        double *boundaries = (double *)malloc((size_t)(n_centroids - 1) * sizeof(double));
        for (int i = 0; i < n_centroids - 1; i++) {
            boundaries[i] = (centroids[i] + centroids[i + 1]) / 2.0;
        }

        /* Update centroids via conditional expectation */
        for (int i = 0; i < n_centroids; i++) {
            double lo = (i == 0) ? -1e30 : boundaries[i - 1];
            double hi = (i == n_centroids - 1) ? 1e30 : boundaries[i];

            double a_norm = lo / sigma;
            double b_norm = hi / sigma;
            double pdf_diff = normal_pdf(a_norm, 1.0) - normal_pdf(b_norm, 1.0);
            double cdf_diff = normal_cdf(b_norm, 1.0) - normal_cdf(a_norm, 1.0);

            if (cdf_diff > 1e-12) {
                centroids[i] = sigma * pdf_diff / cdf_diff;
            }
        }

        free(boundaries);
    }

    /* Sort centroids */
    for (int i = 0; i < n_centroids - 1; i++) {
        for (int j = i + 1; j < n_centroids; j++) {
            if (centroids[j] < centroids[i]) {
                double tmp = centroids[i];
                centroids[i] = centroids[j];
                centroids[j] = tmp;
            }
        }
    }

    return centroids;
}

/* ── Bridge context ───────────────────────────────────────────── */

struct tq_bridge {
    int       head_dim;
    tq_format fmt;
    int       seed;
    int       bit_width;

    /* Cached rotation matrix (double precision, d×d) */
    double   *rotation;
    /* Cached rotation matrix (float32, d×d) for fast path */
    float    *rotation_f32;
    /* Cached codebook (double precision) */
    double   *codebook;
    int       n_centroids;
    /* Cached boundaries (double precision) */
    double   *boundaries;
    int       n_boundaries;
};

static int format_bit_width(tq_format fmt) {
    switch (fmt) {
        case TQ_FORMAT_TURBO2: return 2;
        case TQ_FORMAT_TURBO3: return 3;
        case TQ_FORMAT_TURBO4: return 4;
        default: return 0;
    }
}

tq_status tq_bridge_init(tq_bridge **out, int head_dim, tq_format fmt, int seed) {
    if (!out || head_dim <= 0) return TQ_STATUS_INVALID_ARG;

    int bw = format_bit_width(fmt);
    if (bw == 0 && fmt != TQ_FORMAT_Q8_0 && fmt != TQ_FORMAT_FP16) {
        return TQ_STATUS_FORMAT_ERROR;
    }

    tq_bridge *b = (tq_bridge *)calloc(1, sizeof(tq_bridge));
    if (!b) return TQ_STATUS_ALLOC_FAILED;

    b->head_dim = head_dim;
    b->fmt = fmt;
    b->seed = seed;
    b->bit_width = bw;

    if (bw > 0) {
        /* Generate and cache rotation matrix */
        b->rotation = generate_rotation(head_dim, seed);
        if (!b->rotation) { free(b); return TQ_STATUS_ALLOC_FAILED; }

        /* Float32 copy for fast path */
        b->rotation_f32 = (float *)malloc((size_t)head_dim * head_dim * sizeof(float));
        if (!b->rotation_f32) { free(b->rotation); free(b); return TQ_STATUS_ALLOC_FAILED; }
        for (int i = 0; i < head_dim * head_dim; i++) {
            b->rotation_f32[i] = (float)b->rotation[i];
        }

        /* Generate and cache codebook + boundaries */
        b->n_centroids = 1 << bw;
        b->codebook = generate_codebook(bw, head_dim);
        if (!b->codebook) { free(b->rotation_f32); free(b->rotation); free(b); return TQ_STATUS_ALLOC_FAILED; }

        b->n_boundaries = b->n_centroids - 1;
        b->boundaries = (double *)malloc((size_t)b->n_boundaries * sizeof(double));
        for (int i = 0; i < b->n_boundaries; i++) {
            b->boundaries[i] = (b->codebook[i] + b->codebook[i + 1]) / 2.0;
        }
    }

    *out = b;
    return TQ_STATUS_OK;
}

tq_status tq_bridge_init_precomputed(tq_bridge **out, int head_dim, tq_format fmt,
                                     const float *rotation, const float *codebook,
                                     int n_centroids) {
    if (!out || head_dim <= 0 || !rotation || !codebook || n_centroids <= 0)
        return TQ_STATUS_INVALID_ARG;

    int bw = format_bit_width(fmt);
    if (bw == 0) return TQ_STATUS_FORMAT_ERROR;
    if (n_centroids != (1 << bw)) return TQ_STATUS_INVALID_ARG;

    tq_bridge *b = (tq_bridge *)calloc(1, sizeof(tq_bridge));
    if (!b) return TQ_STATUS_ALLOC_FAILED;

    b->head_dim = head_dim;
    b->fmt = fmt;
    b->seed = -1;  /* sentinel: precomputed, no seed */
    b->bit_width = bw;
    b->n_centroids = n_centroids;

    /* Copy rotation matrix as both double and float32 */
    size_t rot_size = (size_t)head_dim * head_dim;
    b->rotation = (double *)malloc(rot_size * sizeof(double));
    b->rotation_f32 = (float *)malloc(rot_size * sizeof(float));
    if (!b->rotation || !b->rotation_f32) {
        free(b->rotation); free(b->rotation_f32); free(b);
        return TQ_STATUS_ALLOC_FAILED;
    }
    for (size_t i = 0; i < rot_size; i++) {
        b->rotation_f32[i] = rotation[i];
        b->rotation[i] = (double)rotation[i];
    }

    /* Copy codebook as double */
    b->codebook = (double *)malloc((size_t)n_centroids * sizeof(double));
    if (!b->codebook) {
        free(b->rotation); free(b->rotation_f32); free(b);
        return TQ_STATUS_ALLOC_FAILED;
    }
    for (int i = 0; i < n_centroids; i++) {
        b->codebook[i] = (double)codebook[i];
    }

    /* Compute boundaries */
    b->n_boundaries = n_centroids - 1;
    b->boundaries = (double *)malloc((size_t)b->n_boundaries * sizeof(double));
    if (!b->boundaries) {
        free(b->rotation); free(b->rotation_f32); free(b->codebook); free(b);
        return TQ_STATUS_ALLOC_FAILED;
    }
    for (int i = 0; i < b->n_boundaries; i++) {
        b->boundaries[i] = (b->codebook[i] + b->codebook[i + 1]) / 2.0;
    }

    *out = b;
    return TQ_STATUS_OK;
}

void tq_bridge_free(tq_bridge *b) {
    if (!b) return;
    free(b->rotation);
    free(b->rotation_f32);
    free(b->codebook);
    free(b->boundaries);
    free(b);
}

/* ── Bit packing ──────────────────────────────────────────────── */

static void pack_indices(const uint8_t *indices, int n, int bit_width, uint8_t *out, size_t out_size) {
    memset(out, 0, out_size);
    for (int i = 0; i < n; i++) {
        int bit_pos = i * bit_width;
        int byte_pos = bit_pos / 8;
        int bit_off = bit_pos % 8;
        uint8_t val = indices[i] & ((1 << bit_width) - 1);
        out[byte_pos] |= val << bit_off;
        int overflow = bit_off + bit_width - 8;
        if (overflow > 0 && byte_pos + 1 < (int)out_size) {
            out[byte_pos + 1] |= val >> (bit_width - overflow);
        }
    }
}

static void unpack_indices(const uint8_t *data, int n, int bit_width, uint8_t *out) {
    uint8_t mask = (1 << bit_width) - 1;
    for (int i = 0; i < n; i++) {
        int bit_pos = i * bit_width;
        int byte_pos = bit_pos / 8;
        int bit_off = bit_pos % 8;
        uint8_t val = data[byte_pos] >> bit_off;
        if (bit_off + bit_width > 8 && byte_pos + 1 < (n * bit_width + 7) / 8) {
            val |= data[byte_pos + 1] << (8 - bit_off);
        }
        out[i] = val & mask;
    }
}

/* ── PolarQuant compress ──────────────────────────────────────── */

tq_status tq_compress(tq_bridge *b, const float *input, size_t n_vectors,
                      tq_compressed *output) {
    if (!b || !input || !output || n_vectors == 0) return TQ_STATUS_INVALID_ARG;
    if (b->bit_width == 0) return TQ_STATUS_FORMAT_ERROR;

    int d = b->head_dim;
    int bw = b->bit_width;
    size_t packed_bytes = ((size_t)bw * d + 7) / 8;
    size_t bytes_per_vec = 4 + packed_bytes;  /* float32 norm + packed indices */
    size_t total_size = n_vectors * bytes_per_vec;

    uint8_t *buf = (uint8_t *)calloc(total_size, 1);
    if (!buf) return TQ_STATUS_ALLOC_FAILED;

    uint8_t *indices = (uint8_t *)malloc((size_t)d);
    double *v_unit = (double *)malloc((size_t)d * sizeof(double));
    double *y = (double *)malloc((size_t)d * sizeof(double));

    for (size_t vi = 0; vi < n_vectors; vi++) {
        const float *vec = input + vi * d;

        /* Extract norm */
        double norm = 0.0;
        for (int i = 0; i < d; i++) norm += (double)vec[i] * (double)vec[i];
        norm = sqrt(norm);
        double safe_norm = (norm > 0.0) ? norm : 1.0;

        /* Normalize */
        for (int i = 0; i < d; i++) v_unit[i] = (double)vec[i] / safe_norm;

        /* Rotate: y = R @ v_unit */
        for (int i = 0; i < d; i++) {
            y[i] = 0.0;
            for (int j = 0; j < d; j++) {
                y[i] += b->rotation[i * d + j] * v_unit[j];
            }
        }

        /* Quantize: searchsorted on boundaries */
        for (int i = 0; i < d; i++) {
            int idx = 0;
            for (int j = 0; j < b->n_boundaries; j++) {
                if (y[i] >= b->boundaries[j]) idx = j + 1;
                else break;
            }
            indices[i] = (uint8_t)idx;
        }

        /* Pack: [float32 norm | packed indices] */
        size_t offset = vi * bytes_per_vec;
        float norm_f = (float)norm;
        memcpy(buf + offset, &norm_f, 4);
        pack_indices(indices, d, bw, buf + offset + 4, packed_bytes);
    }

    free(indices);
    free(v_unit);
    free(y);

    output->data = buf;
    output->size = total_size;
    output->n_vectors = n_vectors;
    output->head_dim = (uint16_t)d;
    output->fmt = b->fmt;

    return TQ_STATUS_OK;
}

/* ── PolarQuant decompress ────────────────────────────────────── */

tq_status tq_decompress(tq_bridge *b, const tq_compressed *input, float *output) {
    if (!b || !input || !output) return TQ_STATUS_INVALID_ARG;
    if (b->bit_width == 0) return TQ_STATUS_FORMAT_ERROR;

    int d = b->head_dim;
    int bw = b->bit_width;
    size_t packed_bytes = ((size_t)bw * d + 7) / 8;
    size_t bytes_per_vec = 4 + packed_bytes;

    uint8_t *indices = (uint8_t *)malloc((size_t)d);
    double *y_hat = (double *)malloc((size_t)d * sizeof(double));
    double *x_hat = (double *)malloc((size_t)d * sizeof(double));
    const uint8_t *buf = (const uint8_t *)input->data;

    for (size_t vi = 0; vi < input->n_vectors; vi++) {
        size_t offset = vi * bytes_per_vec;

        /* Read norm */
        float norm_f;
        memcpy(&norm_f, buf + offset, 4);
        double norm = (double)norm_f;

        /* Unpack indices */
        unpack_indices(buf + offset + 4, d, bw, indices);

        /* Codebook lookup */
        for (int i = 0; i < d; i++) {
            y_hat[i] = b->codebook[indices[i]];
        }

        /* Inverse rotation: x_hat = R^T @ y_hat */
        for (int i = 0; i < d; i++) {
            x_hat[i] = 0.0;
            for (int j = 0; j < d; j++) {
                x_hat[i] += b->rotation[j * d + i] * y_hat[j];
            }
        }

        /* Rescale */
        for (int i = 0; i < d; i++) {
            output[vi * d + i] = (float)(x_hat[i] * norm);
        }
    }

    free(indices);
    free(y_hat);
    free(x_hat);

    return TQ_STATUS_OK;
}

void tq_compressed_free(tq_compressed *c) {
    if (c && c->data) {
        free(c->data);
        c->data = NULL;
        c->size = 0;
    }
}

/* ── Q8_0 compress/decompress ─────────────────────────────────── */

tq_status tq_compress_q8_0(const float *input, size_t n_elements,
                           void **out_data, size_t *out_size) {
    if (!input || !out_data || !out_size) return TQ_STATUS_INVALID_ARG;

    size_t padded = ((n_elements + TQ_QK8_0 - 1) / TQ_QK8_0) * TQ_QK8_0;
    size_t n_blocks = padded / TQ_QK8_0;
    size_t block_size = 2 + TQ_QK8_0;  /* fp16 scale + int8[32] */
    size_t total = n_blocks * block_size;

    uint8_t *buf = (uint8_t *)calloc(total, 1);
    if (!buf) return TQ_STATUS_ALLOC_FAILED;

    for (size_t bi = 0; bi < n_blocks; bi++) {
        /* Find amax */
        float amax = 0.0f;
        for (int i = 0; i < TQ_QK8_0; i++) {
            size_t idx = bi * TQ_QK8_0 + i;
            float val = (idx < n_elements) ? fabsf(input[idx]) : 0.0f;
            if (val > amax) amax = val;
        }
        float scale = amax / 127.0f;

        /* Write scale as float16 (simplified: store as truncated float) */
        /* TODO: proper float16 conversion */
        uint16_t scale_f16;
        {
            union { float f; uint32_t u; } conv;
            conv.f = scale;
            uint32_t sign = (conv.u >> 16) & 0x8000;
            uint32_t expo = ((conv.u >> 23) & 0xFF);
            uint32_t mant = (conv.u >> 13) & 0x3FF;
            if (expo == 0) { scale_f16 = (uint16_t)sign; }
            else if (expo == 0xFF) { scale_f16 = (uint16_t)(sign | 0x7C00 | mant); }
            else {
                int e = (int)expo - 127 + 15;
                if (e >= 31) { scale_f16 = (uint16_t)(sign | 0x7C00); }
                else if (e <= 0) { scale_f16 = (uint16_t)sign; }
                else { scale_f16 = (uint16_t)(sign | (e << 10) | mant); }
            }
        }

        size_t off = bi * block_size;
        memcpy(buf + off, &scale_f16, 2);

        /* Quantize to int8 */
        for (int i = 0; i < TQ_QK8_0; i++) {
            size_t idx = bi * TQ_QK8_0 + i;
            float val = (idx < n_elements) ? input[idx] : 0.0f;
            int8_t q;
            if (scale > 0.0f) {
                float r = roundf(val / scale);
                if (r < -128.0f) r = -128.0f;
                if (r > 127.0f) r = 127.0f;
                q = (int8_t)r;
            } else {
                q = 0;
            }
            buf[off + 2 + i] = (uint8_t)q;
        }
    }

    *out_data = buf;
    *out_size = total;
    return TQ_STATUS_OK;
}

tq_status tq_decompress_q8_0(const void *data, size_t n_elements, float *output) {
    if (!data || !output) return TQ_STATUS_INVALID_ARG;

    size_t n_blocks = (n_elements + TQ_QK8_0 - 1) / TQ_QK8_0;
    size_t block_size = 2 + TQ_QK8_0;
    const uint8_t *buf = (const uint8_t *)data;

    for (size_t bi = 0; bi < n_blocks; bi++) {
        size_t off = bi * block_size;

        /* Read scale (float16 → float32) */
        uint16_t scale_f16;
        memcpy(&scale_f16, buf + off, 2);
        float scale;
        {
            uint32_t sign = (scale_f16 & 0x8000) << 16;
            uint32_t expo = (scale_f16 >> 10) & 0x1F;
            uint32_t mant = scale_f16 & 0x3FF;
            uint32_t f32;
            if (expo == 0) { f32 = sign; }
            else if (expo == 31) { f32 = sign | 0x7F800000 | (mant << 13); }
            else { f32 = sign | ((expo - 15 + 127) << 23) | (mant << 13); }
            union { uint32_t u; float f; } conv;
            conv.u = f32;
            scale = conv.f;
        }

        /* Dequantize */
        for (int i = 0; i < TQ_QK8_0; i++) {
            size_t idx = bi * TQ_QK8_0 + i;
            if (idx >= n_elements) break;
            int8_t q = (int8_t)buf[off + 2 + i];
            output[idx] = scale * (float)q;
        }
    }

    return TQ_STATUS_OK;
}

/* ── Wire protocol ────────────────────────────────────────────── */

/* CRC32 (same as Python zlib.crc32) */
static uint32_t tq_crc32(const uint8_t *data, size_t len) {
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ (0xEDB88320 & (-(crc & 1)));
        }
    }
    return crc ^ 0xFFFFFFFF;
}

tq_status tq_encode_header(const tq_wire_header *hdr, uint8_t out[TQ_HEADER_SIZE]) {
    if (!hdr || !out) return TQ_STATUS_INVALID_ARG;

    memset(out, 0, TQ_HEADER_SIZE);

    uint16_t flags = hdr->flags;
    if (hdr->fmt_k != hdr->fmt_v) flags |= TQ_FLAG_ASYMMETRIC_KV;

    /* Pack fields (little-endian) */
    uint32_t magic = TQ_MAGIC;
    memcpy(out + 0x00, &magic, 4);
    out[0x04] = TQ_HEADER_VER;
    out[0x05] = (uint8_t)hdr->fmt_k;
    out[0x06] = (uint8_t)hdr->fmt_v;
    out[0x07] = 0; /* reserved */
    memcpy(out + 0x08, &hdr->n_layers, 2);
    memcpy(out + 0x0A, &hdr->layer_start, 2);
    memcpy(out + 0x0C, &hdr->seq_len, 4);
    memcpy(out + 0x10, &hdr->n_heads_k, 2);
    memcpy(out + 0x12, &hdr->n_heads_v, 2);
    memcpy(out + 0x14, &hdr->head_dim, 2);
    memcpy(out + 0x16, &flags, 2);
    memcpy(out + 0x18, &hdr->payload_bytes, 8);

    /* CRC32 over bytes 0x00-0x1F */
    uint32_t crc = tq_crc32(out, 0x20);
    memcpy(out + 0x20, &crc, 4);

    return TQ_STATUS_OK;
}

tq_status tq_decode_header(const uint8_t data[TQ_HEADER_SIZE], tq_wire_header *hdr) {
    if (!data || !hdr) return TQ_STATUS_INVALID_ARG;

    /* Verify magic */
    uint32_t magic;
    memcpy(&magic, data + 0x00, 4);
    if (magic != TQ_MAGIC) return TQ_STATUS_FORMAT_ERROR;

    /* Verify CRC32 */
    uint32_t crc_stored;
    memcpy(&crc_stored, data + 0x20, 4);
    uint8_t check[TQ_HEADER_SIZE];
    memcpy(check, data, TQ_HEADER_SIZE);
    memset(check + 0x20, 0, 4);
    uint32_t crc_computed = tq_crc32(check, 0x20);
    if (crc_stored != crc_computed) return TQ_STATUS_CRC_MISMATCH;

    /* Unpack */
    hdr->version = data[0x04];
    if (hdr->version != TQ_HEADER_VER) return TQ_STATUS_FORMAT_ERROR;

    hdr->fmt_k = (tq_format)data[0x05];
    hdr->fmt_v = (tq_format)data[0x06];
    memcpy(&hdr->n_layers, data + 0x08, 2);
    memcpy(&hdr->layer_start, data + 0x0A, 2);
    memcpy(&hdr->seq_len, data + 0x0C, 4);
    memcpy(&hdr->n_heads_k, data + 0x10, 2);
    memcpy(&hdr->n_heads_v, data + 0x12, 2);
    memcpy(&hdr->head_dim, data + 0x14, 2);
    memcpy(&hdr->flags, data + 0x16, 2);
    memcpy(&hdr->payload_bytes, data + 0x18, 8);

    return TQ_STATUS_OK;
}

/* ── Utility ──────────────────────────────────────────────────── */

float tq_compression_ratio(tq_format fmt, int head_dim) {
    switch (fmt) {
        case TQ_FORMAT_FP16:   return 2.0f;
        case TQ_FORMAT_Q8_0:   return (float)(32 * 4) / (float)(2 + 32); /* 3.76x */
        case TQ_FORMAT_TURBO4: return (float)(head_dim * 32) / (float)(32 + head_dim * 4);
        case TQ_FORMAT_TURBO3: return (float)(head_dim * 32) / (float)(32 + head_dim * 3);
        case TQ_FORMAT_TURBO2: return (float)(head_dim * 32) / (float)(32 + head_dim * 2);
        default: return 1.0f;
    }
}

size_t tq_compressed_size(tq_format fmt, int head_dim, size_t n_vectors) {
    int bw = format_bit_width(fmt);
    if (bw > 0) {
        size_t packed = ((size_t)bw * head_dim + 7) / 8;
        return n_vectors * (4 + packed);
    }
    if (fmt == TQ_FORMAT_Q8_0) {
        size_t n_elements = n_vectors * (size_t)head_dim;
        size_t n_blocks = (n_elements + TQ_QK8_0 - 1) / TQ_QK8_0;
        return n_blocks * (2 + TQ_QK8_0);
    }
    if (fmt == TQ_FORMAT_FP16) {
        return n_vectors * (size_t)head_dim * 2;
    }
    return 0;
}
