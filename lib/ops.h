#ifndef OPS_H_
#define OPS_H_

int *pow(int *d_input, int n, int other);

int *div(int *d_input, int n, int other);

int *mul(int *d_input, int n, int other);

int *sub(int *d_input, int n, int other);

int *sum(int *d_input, int n);

int *sum_cols(int *d_input, int n_cols, int n_rows);

__device__ int parse_int(int *d_input, int start, int space, int pad);

int h_parse_int(char *input, int start);

#endif
