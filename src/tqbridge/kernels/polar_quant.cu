/**
 * polar_quant.cu — Optimized CUDA kernels for PolarQuant compress/decompress.
 *
 * HEAD_DIM and N_BOUNDARIES are #defined at compile time for each format.
 * All parameters are passed as GPU buffers — no scalar kernel arguments,
 * since tinygrad's HCQ dispatch doesn't support raw CUDA scalar params.
 *
 * Layout: 1 block per vector, HEAD_DIM threads per block.
 */

#ifndef HEAD_DIM
#define HEAD_DIM 128
#endif

#ifndef N_BOUNDARIES
#define N_BOUNDARIES 7
#endif

extern "C" {

/**
 * params[0] = n_vectors
 */
__global__ void polar_compress_kernel(
    const float* __restrict__ input,
    const float* __restrict__ rotation,
    const float* __restrict__ boundaries,
    float* __restrict__ norms_out,
    unsigned char* __restrict__ indices_out,
    const int* __restrict__ params
) {
    int vid = blockIdx.x;
    int n_vectors = params[0];
    if (vid >= n_vectors) return;
    int tid = threadIdx.x;

    const float* vec = input + vid * HEAD_DIM;

    /* Load vector into shared memory */
    __shared__ float s_vec[HEAD_DIM];
    __shared__ float s_reduce[HEAD_DIM];
    s_vec[tid] = vec[tid];
    __syncthreads();

    /* Cooperative norm via tree reduction */
    s_reduce[tid] = s_vec[tid] * s_vec[tid];
    __syncthreads();
    for (int stride = HEAD_DIM / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        __syncthreads();
    }

    float norm = sqrtf(s_reduce[0]);
    float inv_norm = (norm > 0.0f) ? (1.0f / norm) : 1.0f;

    if (tid == 0) {
        norms_out[vid] = norm;
    }

    /* Rotate: y[tid] = dot(R[tid], vec * inv_norm) */
    const float* R_row = rotation + tid * HEAD_DIM;
    float y = 0.0f;
    for (int j = 0; j < HEAD_DIM; j++) {
        y += R_row[j] * s_vec[j];
    }
    y *= inv_norm;

    /* Quantize */
    int idx = 0;
    for (int b = 0; b < N_BOUNDARIES; b++) {
        if (y >= boundaries[b]) {
            idx = b + 1;
        } else {
            break;
        }
    }
    indices_out[vid * HEAD_DIM + tid] = (unsigned char)idx;
}

/**
 * params[0] = n_vectors
 */
__global__ void polar_decompress_kernel(
    const float* __restrict__ norms,
    const unsigned char* __restrict__ indices,
    const float* __restrict__ rotation,
    const float* __restrict__ codebook,
    float* __restrict__ output,
    const int* __restrict__ params
) {
    int vid = blockIdx.x;
    int n_vectors = params[0];
    if (vid >= n_vectors) return;
    int tid = threadIdx.x;

    const unsigned char* idx = indices + vid * HEAD_DIM;

    /* Load codebook-looked-up values into shared memory */
    __shared__ float s_y_hat[HEAD_DIM];
    s_y_hat[tid] = codebook[idx[tid]];
    __syncthreads();

    /* Inverse rotation: x_hat[tid] = dot(column tid of R, y_hat) */
    float val = 0.0f;
    for (int j = 0; j < HEAD_DIM; j++) {
        val += rotation[j * HEAD_DIM + tid] * s_y_hat[j];
    }

    output[vid * HEAD_DIM + tid] = val * norms[vid];
}

} /* extern "C" */
