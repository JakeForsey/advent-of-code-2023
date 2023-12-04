#include <stdio.h>

#include "../../lib/aoc.h"
#include "../../lib/cuda.h"
#include "../../lib/ops.h"

int c_parse_int(char *input, int start, int space, int pad) {
    int digits[7];  // Parse at most 7 digits
    int n_digits = 0;
    for (int i = 0; i < 7; i++) {
        int c = input[start + i];
        if (c == space | c == pad | c == 10) {
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

__global__ void duplicates(int *d_input, int n_cols, int n_rows, int *d_out) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row < n_rows && col < n_cols) {
        int value = d_input[(row * n_cols) + col];
        int match = 0;
        for (int i = 0; i < n_cols; i++) {
            if (i == col) {
                continue;
            }
            if (d_input[(row * n_cols) + i] == value) {
                match = 1;
            }
        }
        d_out[(row * n_cols) + col] = match;
    }
}

void part1(int *d_input, int n_cols, int n_rows) {
    int *d_wins = empty(n_cols * n_rows);
    duplicates<<<n_rows, n_cols>>>(d_input, n_cols, n_rows, d_wins);
    int *d_total_wins = div(sum_cols(d_wins, n_cols, n_rows), n_rows, 2);
    int *d_scores = pow(sub(d_total_wins, n_rows, 1), n_rows, 2);
    printf("part1: %d\n", from_device(sum(d_scores, n_rows), 1)[0]);
}

int main() {
    char *text = read_file((char*) "days/day04/input");
    int input[100000];
    int j = 0; // Position in input
    int i = 0; // Position in text (skip Card N)
    int col = 0;
    int row = 0;
    int n_cols = 0;
    while (true) {
        if (i >= strlen(text)) {
            break;
        }
        int value = text[i];
        if (value == 67) {
            // Handle "G" from "Card N:"
            while (text[i] != 58) {
                // Skip to ":"
                i += 1;
            }
            i += 1;
        } else if (value == 10) {
            // Handle new line
            if (col > n_cols) {
                n_cols = col;
            }
            col = 0;
            row += 1;
            i += 1;
        } else if (value == 32) {
            // Handle "space"
            i += 1;
        } else if (value == 124) {
            // Handle "|"
            i += 1;
        } else {
            // Otherwise parse the integer
            int a = c_parse_int(text, i, 32, 0);
            input[j] = a;
            i += a > 9 ? 2 : 1;  // Hhandle 1 and 2 digit numbers
            j += 1;
            col += 1;
        }
    }
    int n_rows = row + 1;
    int *d_input = to_device(input, n_rows * n_cols);

    part1(d_input, n_cols, n_rows);

    return 0;
}
