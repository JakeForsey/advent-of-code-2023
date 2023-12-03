#ifndef OPS_H_
#define OPS_H_

int *mul(int *d_input, int n, int other);

int *sum(int *d_input, int n);

__device__ int parse_int(int *d_input, int start, int space, int pad);

#endif
