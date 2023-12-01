#include "cuda.h"
#include "ops.h"

// Ops

__device__ int _mul_op(int d_a, int d_b) {
    return d_a * d_b;
}

__device__ int _sum_op(int d_a, int d_b) {
    return d_a + d_b;
}

// Kernels

__global__ void _constant(int op, int *d_input, int n, int d_other, int *d_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (op == 0) {
            d_out[i] = _sum_op(d_input[i], d_other);
        } else if (op == 1) {
            d_out[i] = _mul_op(d_input[i], d_other);
        }
    }
}

__global__ void _reduce_par(int op, int *d_input, int n, int *d_out) {
    extern __shared__ int block_memory[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_idx = threadIdx.x;
    
    block_memory[thread_idx] = d_input[i];
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (thread_idx % (2 * stride) == 0) {
            if (op == 0) {
                block_memory[thread_idx] = _sum_op(block_memory[thread_idx], block_memory[thread_idx + stride]);
            } else if (op == 1) {
                block_memory[thread_idx] = _mul_op(block_memory[thread_idx], block_memory[thread_idx + stride]);
            }
        }
        __syncthreads();
    }

    if (thread_idx == 0) {
        d_out[blockIdx.x] = block_memory[0];
    }
}

__global__ void _reduce_seq(int op, int *d_input, int n, int *d_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        for (int j = 0; j < n; j++) {
            if (op == 0) {
                d_out[0] = _sum_op(d_out[0], d_input[j]);
            } else if (op == 1) {
                d_out[0] = _mul_op(d_out[0], d_input[j]);
            }
        }
    }
}

// Dispatch

int *_constant(int op, int *d_input, int n, int d_other) {
    int *d_out = empty(n);
    _constant<<<blocks(n), threads(n)>>>(op, d_input, n, d_other, d_out);
    return d_out;
}

int *_reduce(int op, int *d_input, int n) {
    int threads = 512 > n ? n + 1 : 512;
    int blocks = ceil((double) (n + 1) / (double) threads);
    int *d_out = empty(blocks);
    _reduce_par<<<blocks, threads, threads * sizeof(int)>>>(op, d_input, n, d_out);
    if (blocks > 1) {
        int *d_out2 = empty(1);
        _reduce_seq<<<1, 1>>>(op, d_out, blocks, d_out2);
        return d_out2;
    } else {
        return d_out;
    }
}

// Interface

int *mul(int *d_input, int n, int other) {
    return _constant(1, d_input, n, other);
}

int *sum(int *d_input, int n) {
    return _reduce(0, d_input, n);
}
