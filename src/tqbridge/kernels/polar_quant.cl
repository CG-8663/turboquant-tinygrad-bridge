/**
 * polar_quant.cl — OpenCL kernels for PolarQuant compress/decompress.
 *
 * Supports AMD integrated GPUs (Radeon, RDNA), Intel Arc/UHD/Iris,
 * and any OpenCL 1.2+ device including CPU-based OpenCL runtimes.
 *
 * For integrated GPUs (iGPU) that share system memory with the CPU,
 * there's no DMA overhead — the GPU reads directly from host memory.
 * This makes iGPUs ideal for TQBridge decode nodes.
 *
 * HEAD_DIM and N_BOUNDARIES are injected via #define at compile time.
 */

#ifndef HEAD_DIM
#define HEAD_DIM 128
#endif

#ifndef N_BOUNDARIES
#define N_BOUNDARIES 7
#endif

/**
 * PolarQuant compress.
 * work_group = 1 per vector, HEAD_DIM work items per group.
 */
__kernel void polar_compress_kernel(
    __global const float* restrict input,
    __global const float* restrict rotation,
    __global const float* restrict boundaries,
    __global float* restrict norms_out,
    __global uchar* restrict indices_out,
    __global const int* restrict params
) {
    int vid = get_group_id(0);
    int n_vectors = params[0];
    if (vid >= n_vectors) return;
    int tid = get_local_id(0);

    __global const float* vec = input + vid * HEAD_DIM;

    /* Load vector into local memory */
    __local float s_vec[HEAD_DIM];
    __local float s_reduce[HEAD_DIM];
    s_vec[tid] = vec[tid];
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Cooperative norm via tree reduction */
    s_reduce[tid] = s_vec[tid] * s_vec[tid];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = HEAD_DIM / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float norm = sqrt(s_reduce[0]);
    float inv_norm = (norm > 0.0f) ? (1.0f / norm) : 1.0f;

    if (tid == 0) {
        norms_out[vid] = norm;
    }

    /* Rotate: y[tid] = dot(R[tid], vec * inv_norm) */
    __global const float* R_row = rotation + tid * HEAD_DIM;
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
    indices_out[vid * HEAD_DIM + tid] = (uchar)idx;
}

/**
 * PolarQuant decompress.
 */
__kernel void polar_decompress_kernel(
    __global const float* restrict norms,
    __global const uchar* restrict indices,
    __global const float* restrict rotation,
    __global const float* restrict codebook,
    __global float* restrict output,
    __global const int* restrict params
) {
    int vid = get_group_id(0);
    int n_vectors = params[0];
    if (vid >= n_vectors) return;
    int tid = get_local_id(0);

    __global const uchar* idx = indices + vid * HEAD_DIM;

    /* Load codebook-looked-up values into local memory */
    __local float s_y_hat[HEAD_DIM];
    s_y_hat[tid] = codebook[idx[tid]];
    barrier(CLK_LOCAL_MEM_FENCE);

    /* Inverse rotation */
    float val = 0.0f;
    for (int j = 0; j < HEAD_DIM; j++) {
        val += rotation[j * HEAD_DIM + tid] * s_y_hat[j];
    }

    output[vid * HEAD_DIM + tid] = val * norms[vid];
}
