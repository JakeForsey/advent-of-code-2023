#ifndef CUDA_H_
#define CUDA_H_

int *empty(int n);

long *empty_long(int n);

int *to_device(int *input, int n);

long *to_device_long(long *input, int n);

int *from_device(int *d_input, int n);

void print_darray(int *d_input, int n);

int threads(int n);

int blocks(int n);

#endif
