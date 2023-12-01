#include <stdio.h>

#include "../../lib/aoc.h"
#include "../../lib/cuda.h"
#include "../../lib/ops.h"

__device__ int match(int *d_input, int input_start, int *d_pattern, int pattern_start, int n, int pad) {
    if (d_input[input_start] == pad) {
        return 0;  // Input is empty...
    }
    if (d_pattern[pattern_start] == pad) {
        return 0;  // Pattern is empty...
    }
    for (int i = 0; i < n; i++) {
        if (d_pattern[pattern_start + i] == pad) {
            return 1;  // Reached the end of the pattern
        }
        if (d_input[input_start + i] != d_pattern[pattern_start + i]) {
            return 0;  // Pattern doesnt match
        }
    }
    return 1;
}

__global__ void first_match(int *d_input, int n_rows, int n_cols, int *d_patterns, int p_rows, int p_cols, int *d_out) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n_rows) {
        int row_matched = 0;
        for (int col = 0; col < n_cols; col++) {
            int remaining = min(p_cols, n_cols - col);
            for (int p_row = 0; p_row < p_rows; p_row++) {
                if (match(d_input, (row * n_cols) + col, d_patterns, (p_row * p_cols), remaining, 0) == 1) {
                    d_out[row] = p_row;
                    row_matched = 1;
                    break;
                }
            }
            if (row_matched == 1) {
                break;
            }
        }
    }
}

__global__ void last_match(int *d_input, int n_rows, int n_cols, int *d_patterns, int p_rows, int p_cols, int *d_out) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n_rows) {
        for (int col = 0; col < n_cols; col++) {
            int remaining = min(p_cols, n_cols - col);
            for (int p_row = 0; p_row < p_rows; p_row++) {
                if (match(d_input, (row * n_cols) + col, d_patterns, (p_row * p_cols), remaining, 0) == 1) {
                    d_out[row] = p_row;
                }
            }
        }
    }
}

__global__ void map(int *d_input, int n, int *d_map, int *d_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d_out[i] = d_map[d_input[i]];
    }
}

int score(int *d_input, int n_rows, int n_cols, int *d_patterns, int p_rows, int p_cols, int*d_map) {
    int *d_first = empty(n_rows);
    first_match<<<blocks(n_rows), threads(n_rows)>>>(d_input, n_rows, n_cols, d_patterns, p_rows, p_cols, d_first);
    map<<<blocks(n_rows), threads(n_rows)>>>(d_first, n_rows, d_map, d_first);
    int *d_first_10 = mul(d_first, n_rows, 10);

    int *d_last = empty(n_rows);
    last_match<<<blocks(n_rows), threads(n_rows)>>>(d_input, n_rows, n_cols, d_patterns, p_rows, p_cols, d_last);
    map<<<blocks(n_rows), threads(n_rows)>>>(d_last, n_rows, d_map, d_last);

    int total = 0;
    total += from_device(sum(d_first_10, n_rows), 1)[0];
    total += from_device(sum(d_last, n_rows), 1)[0];

    return total;
}

void part1(int *d_input, int n_rows, int n_cols) {
    int p_rows = 9;
    int p_cols = 1;

    // ASCII code
    int patterns[p_rows * p_cols] = {
        49,  // 1
        50,  // 2
        51,
        52,
        53,  // ..
        54,
        55,
        56,
        57,  // 9
    };
    int map[p_rows] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    int *d_patterns = to_device(patterns, p_rows * p_cols);
    int *d_map = to_device(map, p_rows);

    printf("part1: %d\n", score(d_input, n_rows, n_cols, d_patterns, p_rows, p_cols, d_map));
}

void part2(int *d_input, int n_rows, int n_cols) {
    int p_rows = 18;
    int p_cols = 6;

    // ASCII code
    int patterns[p_rows * p_cols] = {
        49, 0, 0, 0, 0, 0,  // 1
        50, 0, 0, 0, 0, 0,  // 2
        51, 0, 0, 0, 0, 0,
        52, 0, 0, 0, 0, 0,
        53, 0, 0, 0, 0, 0,  // ..
        54, 0, 0, 0, 0, 0,
        55, 0, 0, 0, 0, 0,
        56, 0, 0, 0, 0, 0,
        57, 0, 0, 0, 0, 0,  // 9
        111, 110, 101, 0, 0, 0,      // one
        116, 119, 111, 0, 0, 0,      // two
        116, 104, 114, 101, 101, 0,  // three
        102, 111, 117, 114, 0, 0,    // four
        102, 105, 118, 101, 0, 0,    // five
        115, 105, 120, 0, 0, 0,      // six
        115, 101, 118, 101, 110, 0,  // seven
        101, 105, 103, 104, 116, 0,  // eight
        110, 105, 110, 101, 0, 0,    // nine
    };
    int map[p_rows] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    int *d_patterns = to_device(patterns, p_rows * p_cols);
    int *d_map = to_device(map, p_rows);

    printf("part2: %d\n", score(d_input, n_rows, n_cols, d_patterns, p_rows, p_cols, d_map));
}

int main() {
    char *text = read_file((char*) "days/day01/input");
    int max_rows = 10000;
    int n_cols = 100;
    int input[max_rows * n_cols];
    int row = 0;
    int col = 0;
    for (int i = 0; i < strlen(text); i++) {
        if (text[i] == 10) {
            row += 1;
            col = 0;
            continue;
        }
        input[(row * n_cols) + col] = text[i];
        col += 1;
    }
    int n_rows = row + 1;

    int *d_input = to_device(input, n_rows * n_cols);
    part1(d_input, n_rows, n_cols);
    part2(d_input, n_rows, n_cols);

    return 0;
}
