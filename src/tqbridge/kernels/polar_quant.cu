/**
 * polar_quant.cu — CUDA kernels for PolarQuant compress/decompress.
 *
 * These kernels run the hot path on GPU, eliminating the CPU round-trip
 * that the native C backend requires. The rotation matrix and codebook
 * are kept in GPU memory.
 *
 * Kernel layout:
 *   compress: 1 thread per vector (each thread does rotation + quantization)
 *   decompress: 1 thread per vector (each thread does lookup + inverse rotation)
 *
 * For head_dim=128, each thread does 128x128 dot products — compute-bound,
 * so parallelism across vectors gives the speedup.
 */

extern "C" {

/**
 * PolarQuant compress: norm extraction → rotation → quantization → pack norms + indices.
 *
 * Args:
 *   input:      (n_vectors, head_dim) float32, source vectors
 *   rotation:   (head_dim, head_dim) float32, precomputed orthogonal rotation matrix
 *   boundaries: (n_boundaries,) float32, quantization boundaries
 *   norms_out:  (n_vectors,) float32, extracted norms
 *   indices_out:(n_vectors, head_dim) uint8, quantized indices
 *   n_vectors:  number of vectors
 *   head_dim:   dimension per vector
 *   n_boundaries: number of quantization boundaries (2^bit_width - 1)
 */
__global__ void polar_compress_kernel(
    const float* __restrict__ input,
    const float* __restrict__ rotation,
    const float* __restrict__ boundaries,
    float* __restrict__ norms_out,
    unsigned char* __restrict__ indices_out,
    int n_vectors,
    int head_dim,
    int n_boundaries
) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= n_vectors) return;

    const float* vec = input + vid * head_dim;

    // Step 1: Extract norm
    float norm_sq = 0.0f;
    for (int i = 0; i < head_dim; i++) {
        norm_sq += vec[i] * vec[i];
    }
    float norm = sqrtf(norm_sq);
    float safe_norm = (norm > 0.0f) ? norm : 1.0f;
    norms_out[vid] = norm;

    // Step 2: Rotate (y = R @ (vec / norm))
    // Step 3: Quantize (searchsorted on boundaries)
    unsigned char* idx_out = indices_out + vid * head_dim;
    for (int i = 0; i < head_dim; i++) {
        // Compute rotated component: dot(rotation[i], vec/norm)
        float y = 0.0f;
        const float* R_row = rotation + i * head_dim;
        for (int j = 0; j < head_dim; j++) {
            y += R_row[j] * (vec[j] / safe_norm);
        }

        // Quantize: find bucket via linear scan of boundaries
        int idx = 0;
        for (int b = 0; b < n_boundaries; b++) {
            if (y >= boundaries[b]) idx = b + 1;
            else break;
        }
        idx_out[i] = (unsigned char)idx;
    }
}

/**
 * PolarQuant decompress: codebook lookup → inverse rotation → rescale.
 *
 * Args:
 *   norms:      (n_vectors,) float32, norms from compress
 *   indices:    (n_vectors, head_dim) uint8, quantized indices
 *   rotation:   (head_dim, head_dim) float32, rotation matrix (uses R^T)
 *   codebook:   (n_centroids,) float32, codebook values
 *   output:     (n_vectors, head_dim) float32, reconstructed vectors
 *   n_vectors:  number of vectors
 *   head_dim:   dimension per vector
 */
__global__ void polar_decompress_kernel(
    const float* __restrict__ norms,
    const unsigned char* __restrict__ indices,
    const float* __restrict__ rotation,
    const float* __restrict__ codebook,
    float* __restrict__ output,
    int n_vectors,
    int head_dim
) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= n_vectors) return;

    float norm = norms[vid];
    const unsigned char* idx = indices + vid * head_dim;
    float* out = output + vid * head_dim;

    // Inverse rotation: x_hat[i] = sum_j(R[j][i] * codebook[idx[j]])
    // This is R^T @ y_hat, computed as: for each output dim i,
    // dot product of column i of R with codebook-looked-up values.
    for (int i = 0; i < head_dim; i++) {
        float val = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            val += rotation[j * head_dim + i] * codebook[idx[j]];
        }
        out[i] = val * norm;
    }
}

} // extern "C"
