#include <stdio.h>

#include "../../lib/aoc.h"
#include "../../lib/cuda.h"
#include "../../lib/ops.h"

__device__ int parse_int(int *d_input, int start, int space, int pad) {
    int digits[5];
    int n_digits = 0;
    for (int i = 0; i < 6; i++) {
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

__global__ void part1(int *d_input, int n_rows, int *d_out) {
    int n_cols = 300;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int start = row * n_cols;

    int r_max = 12;
    int g_max = 13;
    int b_max = 14;

    int r = 114;
    int g = 103;
    int b = 98;

    int space = 32;
    int colon = 58;

    int curr_count = 0;
    int valid = 1;

    if (row < n_rows) {
        int state = 0; 
        int i = start;
        while (i < start + 200) {
            if (d_input[i] == 0) {
                // Into the padding on a row...
                break;
            }

            if (state == 0) {
                // Move past the "Game N:"
                if (d_input[i] == colon) {
                    state = 1;
                    i += 1;
                } else {
                    i += 1;
                }

            } else if (state == 1) {
                // Parse a number
                if (d_input[i] != space) {
                    i += 1;
                } else {
                    curr_count = parse_int(d_input, i + 1, space, 0);
                    i += 1;
                    state = 2; // Parse colour
                }

            } else if (state == 2) {
                // Parse a colour
                if (d_input[i] == r) {
                    if (curr_count > r_max) {
                        valid = 0;
                        break;
                    }
                    i += 3;
                    state = 1;  // Parse number
                } else if (d_input[i] == g) {
                    if (curr_count > g_max) {
                        valid = 0;
                        break;
                    }
                    i += 3;
                    state = 1;  // Parse number
                } else if (d_input[i] == b) {
                    if (curr_count > b_max) {
                        valid = 0;
                        break;
                    }
                    i += 3;
                    state = 1;  // Parse number
                }else {
                    i += 1;
                }
            }
        }
        if (valid == 1) {
            // d_out[row] = curr_count;
            d_out[row] = row + 1;
        }
    }
}

__global__ void part2(int *d_input, int n_rows, int *d_out) {
    int n_cols = 300;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int start = row * n_cols;

    int r_min = 0;
    int g_min = 0;
    int b_min = 0;

    int r = 114;
    int g = 103;
    int b = 98;

    int space = 32;
    int colon = 58;

    int curr_count = 0;
    int valid = 1;

    if (row < n_rows) {
        int state = 0; 
        int i = start;
        while (i < start + 200) {
            if (d_input[i] == 0) {
                // Into the padding on a row...
                break;
            }

            if (state == 0) {
                // Move past the "Game N:"
                if (d_input[i] == colon) {
                    state = 1;
                    i += 1;
                } else {
                    i += 1;
                }

            } else if (state == 1) {
                // Parse a number
                if (d_input[i] != space) {
                    i += 1;
                } else {
                    curr_count = parse_int(d_input, i + 1, space, 0);
                    i += 1;
                    state = 2; // Parse colour
                }

            } else if (state == 2) {
                // Parse a colour
                if (d_input[i] == r) {
                    if (curr_count > r_min) {
                        r_min = curr_count;
                    }
                    i += 3;
                    state = 1;  // Parse number
                } else if (d_input[i] == g) {
                    if (curr_count > g_min) {
                        g_min = curr_count;
                    }
                    i += 3;
                    state = 1;  // Parse number
                } else if (d_input[i] == b) {
                    if (curr_count > b_min) {
                        b_min = curr_count;
                    }
                    i += 3;
                    state = 1;  // Parse number
                } else {
                    i += 1;
                }
            }
        }
        if (valid == 1) {
            d_out[row] = r_min * g_min * b_min;
        }
    }
}

void part1(int* d_input, int n_rows) {
    int *d_out = empty(n_rows);
    part1<<<blocks(n_rows), threads(n_rows)>>>(d_input, n_rows, d_out);
    int *part1_result = sum(d_out, n_rows);
    printf("part1: %d\n", from_device(part1_result, 1)[0]);
}

void part2(int* d_input, int n_rows) {
    int *d_out = empty(n_rows);
    part2<<<blocks(n_rows), threads(n_rows)>>>(d_input, n_rows, d_out);
    int *part1_result = sum(d_out, n_rows);
    printf("part2: %d\n", from_device(part1_result, 1)[0]);
}

int main() {
    char *text = read_file((char*) "days/day02/input");
    int max_rows = 100;
    int n_cols = 300;
    int input[max_rows * n_cols];
    int row = 0;
    int col = 0;
    for (int i = 0; i < strlen(text); i++) {
        if (text[i] == 10) {
            row += 1;
            col = 0;
            continue;
        }
        input[(row * n_cols) + col] = (int) text[i];
        col += 1;
    }
    int n_rows = row + 1;
    int *d_input = to_device(input, n_rows * n_cols);

    part1(d_input, n_rows);
    part2(d_input, n_rows);

    return 0;
}
