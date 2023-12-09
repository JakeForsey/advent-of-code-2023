#include <stdio.h>

#include "../../lib/aoc.h"
#include "../../lib/cuda.h"
#include "../../lib/ops.h"

int characters(int n) {
    int chars = 0;
    if (n < 0) {
        n = -n;
        chars += 1;
    }
    if (n < 10) {
        return chars + 1;
    } else if (n < 100) {
        return chars + 2;
    } else if (n < 1000) {
        return chars + 3;
    } else if (n < 10000) {
        return chars + 4;
    } else if (n < 100000) {
        return chars + 5;
    } else if (n < 1000000) {
        return chars + 6;
    } else if (n < 10000000) {
        return chars + 7;
    } else if (n < 100000000) {
        return chars + 8;
    }
    return chars;
}

__global__ void predict(int *d_input, int n_rows, int n_cols, int *d_out) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int max_cols = 25;
    int max_depth = 25;
    if (row < n_rows) {
        int diffs[25 * 25];  // constant max_cols * max_depth

        // Put initial series into diffs
        for (int col = 0; col < n_cols; col ++) {
            int value = d_input[(max_cols * row) + col];
            diffs[col] = value;
        }

        // Calculate the diffs
        for (int depth = 1; depth < max_depth; depth++) {
            int depth_offset = (depth - 1) * max_cols;
            for (int col = 0; col < (max_cols - depth); col++) {
                int n = diffs[depth_offset + col + 1] - diffs[depth_offset + col];
                diffs[(depth * max_cols) + col] = n;
            }
        }

        // Sum up all the tails of the diffs
        for (int depth = max_depth; depth > 0; depth--) {
            int bottom = (depth * max_cols) + n_cols - depth;
            int top = ((depth - 1) * max_cols) + n_cols - 1 - (depth - 1);
            diffs[top + 1] = diffs[top] + diffs[bottom];
        }
        d_out[row] = diffs[n_cols];
    }
}

void part1(int *d_input, int n_rows, int n_cols) {
    int *d_out = empty(n_rows);
    predict<<<blocks(n_rows), threads(n_rows)>>>(d_input, n_rows, n_cols, d_out);
    printf("part1: %d\n", from_device(sum(d_out, n_rows), 1)[0]);
}

int main() {
    char *text = read_file((char*) "days/day09/input");

    int max_rows = 200;
    int max_cols = 25;
    int input[max_rows * max_cols];

    int row = 0;
    int col = 0;
    int n_cols = 0;
    for (int i = 0; i < strlen(text) + 1; i++) {
        int value = text[i];
        if (value == 10 | i == strlen(text)) {
            // New line
            n_cols = col;
            row += 1;
            col = 0;
            continue;
        } else if (value == 32) {
            // Space
            continue;
        }
        int n = h_parse_int(text, i);
        input[(max_cols * row) + col] = n;
        i += characters(n) - 1;
        col += 1;
    }
    int n_rows = row;
    int *d_input = to_device(input, n_rows * max_cols);
    part1(d_input, n_rows, n_cols);
    return 0;
}
