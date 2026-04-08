// polar_quant.metal — Metal compute shaders for PolarQuant compress/decompress.
// HEAD_DIM and N_BOUNDARIES are injected before this file via string prepend.
#include <metal_stdlib>
using namespace metal;

kernel void polar_compress_kernel(
    device const float* input [[buffer(0)]],
    device const float* rotation [[buffer(1)]],
    device const float* boundaries [[buffer(2)]],
    device float* norms_out [[buffer(3)]],
    device uchar* indices_out [[buffer(4)]],
    device const int* params [[buffer(5)]],
    uint vid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    int n_vectors = params[0];
    if ((int)vid >= n_vectors) return;

    device const float* vec = input + vid * HEAD_DIM;

    threadgroup float s_vec[HEAD_DIM];
    threadgroup float s_reduce[HEAD_DIM];
    s_vec[tid] = vec[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    s_reduce[tid] = s_vec[tid] * s_vec[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int stride = HEAD_DIM / 2; stride > 0; stride >>= 1) {
        if ((int)tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float norm = sqrt(s_reduce[0]);
    float inv_norm = (norm > 0.0f) ? (1.0f / norm) : 1.0f;
    if (tid == 0) {
        norms_out[vid] = norm;
    }

    device const float* R_row = rotation + tid * HEAD_DIM;
    float y = 0.0f;
    for (int j = 0; j < HEAD_DIM; j++) {
        y += R_row[j] * s_vec[j];
    }
    y *= inv_norm;

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

kernel void polar_decompress_kernel(
    device const float* norms [[buffer(0)]],
    device const uchar* indices [[buffer(1)]],
    device const float* rotation [[buffer(2)]],
    device const float* codebook [[buffer(3)]],
    device float* output [[buffer(4)]],
    device const int* params [[buffer(5)]],
    uint vid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    int n_vectors = params[0];
    if ((int)vid >= n_vectors) return;

    device const uchar* idx = indices + vid * HEAD_DIM;

    threadgroup float s_y_hat[HEAD_DIM];
    s_y_hat[tid] = codebook[idx[tid]];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float val = 0.0f;
    for (int j = 0; j < HEAD_DIM; j++) {
        val += rotation[j * HEAD_DIM + tid] * s_y_hat[j];
    }

    output[vid * HEAD_DIM + tid] = val * norms[vid];
}
