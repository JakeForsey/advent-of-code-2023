#include "cuda.h"
#include "ops.h"

// Ops

__device__ int _mul_op(int d_a, int d_b) {
    return d_a * d_b;
}

__device__ int _sum_op(int d_a, int d_b) {
    return d_a + d_b;
}

__device__ int _sub_op(int d_a, int d_b) {
    return d_a - d_b;
}

__device__ int _div_op(int d_a, int d_b) {
    return d_a / d_b;
}

// Kernels

__global__ void _constant(int op, int *d_input, int n, int d_other, int *d_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int a = d_input[i];
        int b = d_other;
        if (op == 0) {
            d_out[i] = _sum_op(a, b);
        } else if (op == 1) {
            d_out[i] = _mul_op(a, b);
        } else if (op == 2) {
            d_out[i] = _div_op(a, b);
        } else if (op == 3) {
            d_out[i] = powf(b, a); // eww
        } else if (op == 4) {
            d_out[i] = _sub_op(a, b);
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
            int a = block_memory[thread_idx];
            int b = block_memory[thread_idx + stride];
            if (op == 0) {
                block_memory[thread_idx] = _sum_op(a, b);
            } else if (op == 1) {
                block_memory[thread_idx] = _mul_op(a, b);
            } else if (op == 2) {
                block_memory[thread_idx] = _mul_op(a, b);
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
            int a = d_out[0];
            int b = d_input[j];
            if (op == 0) {
                d_out[0] = _sum_op(a, b);
            } else if (op == 1) {
                d_out[0] = _mul_op(a, b);
            } else if (op == 2) {
                d_out[0] = _div_op(a, b);
            }
        }
    }
}

__global__ void _reduce_cols_seq(int op, int *d_input, int n_cols, int n_rows, int *d_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: Make parallel
    if (i == 0) {
        for (int row = 0; row < n_rows; row++) {
            for (int col = 0; col < n_cols; col++) {
                int a = d_out[row];
                int b = d_input[(row * n_cols) + col];
                if (op == 0) {
                    d_out[row] = _sum_op(a, b);
                } else if (op == 1) {
                    d_out[row] = _mul_op(a, b);
                } else if (op == 2) {
                    d_out[row] == _div_op(a, b);
                }
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

int *_reduce_cols(int op, int *d_input, int n_cols, int n_rows) {
    int *d_out = empty(n_rows);
    _reduce_cols_seq<<<1, 1>>>(op, d_input, n_cols, n_rows, d_out);
    return d_out;
}

// Interface

int *pow (int *d_input, int n, int other) {
    return _constant(3, d_input, n, other);
}

int *div(int *d_input, int n, int other) {
    return _constant(2, d_input, n, other);
}

int *mul(int *d_input, int n, int other) {
    return _constant(1, d_input, n, other);
}

int *sub(int *d_input, int n, int other) {
    return _constant(4, d_input, n, other);
}

int *sum(int *d_input, int n) {
    return _reduce(0, d_input, n);
}

int *sum_cols(int *d_input, int n_cols, int n_rows) {
    return _reduce_cols(0, d_input, n_cols, n_rows);
}

__device__ int parse_int(int *d_input, int start, int space, int pad) {
    int digits[7];  // Parse at most 7 digits
    int n_digits = 0;
    for (int i = 0; i < 7; i++) {
        int c = d_input[start + i];
        if (c == space | c == pad) {
            break;
        }
        n_digits += 1;
        digits[i] = c - 48;
    }
    int out = 0;
    for (int j = 0; j < n_digits; j++) {
        out += pow(10, (n_digits - j - 1)) * digits[j];
    }
    return out;
}
