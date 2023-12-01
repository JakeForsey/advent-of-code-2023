#include <stdio.h>

#include "cuda.h"

int *empty(int n) {
    int *d_input;
    cudaMalloc((void**)&d_input, sizeof(int) * n);
    return d_input;
}

int *to_device(int *input, int n) {
    int *d_input = empty(n);
    cudaMemcpy(d_input, input, sizeof(int) * n, cudaMemcpyHostToDevice);
    return d_input;
}

int *from_device(int *d_input, int n) {
    int *input = (int*)malloc(sizeof(int) * n);
    cudaMemcpy(input, d_input, sizeof(int) * n, cudaMemcpyDeviceToHost);
    return input;
}

int threads(int n) {
    return 512 > n ? n + 1 : 512;
}

int blocks(int n) {
    return ceil((double) (n + 1) / (double) threads(n));
}

void print_darray(int *d_input, int n) {
    int *input = from_device(d_input, n);
    printf("[");
    for (int i = 0; i < n; i++) {
        printf("%d, ", input[i]);
    }
    printf("]\n");
}
