#include <stdio.h>

#include "../../lib/aoc.h"
#include "../../lib/cuda.h"
#include "../../lib/ops.h"

bool is_digit(int a) {
    return a >= 48 && a <= 57;
}

__device__ bool d_is_digit(int a) {
    return a >= 48 && a <= 57;
}

__global__ void process_symbol(int *d_grid, int n_cols, int n_rows, int *d_symbols, int n_symbols, int *d_out) {
    int si = blockIdx.x * blockDim.x + threadIdx.x;
    // Adjacency lookup [x offset, y offset]
    int adj[8][2] = {
        // Cardinals
        {0, 1},
        {0, -1},
        {1, 0},
        {-1, 0},
        // Diagonals
        {1, 1},
        {-1, -1},
        {1, -1},
        {-1, 1},
    };

    int visited[9][9];

    if (si < n_symbols) {
        int sx = d_symbols[(si * 3) + 1];
        int sy = d_symbols[(si * 3) + 2];

        for (int i = 0; i < 8; i++) {
            // Search all the adjacent positions
            int dx = adj[i][0];
            int dy = adj[i][1];
            int x = dx + sx;
            int y = dy + sy;

            if (x < 0 | x > n_cols) {
                continue;  // Outside of the grid
            }
            if (visited[4 + dx][4 + dy] == 1) {
                continue;  // Already visited this grid cell
            }
            visited[4 + dx][4 + dy] = 1;  // Mark as visited

            int value = d_grid[(y * n_cols) + x];
            if (d_is_digit(value)) {
                // Accumulate all the digits to the left and right
                int digits[9] = {
                    0, 0, 0,
                    0, 0, 0,
                    0, 0, 0,
                };
                int start = 4;
                digits[start] = value;

                // Gather digits from the left
                int lx = x;
                int lstart = start;
                while (lx >= 0 && d_is_digit(d_grid[(y * n_cols) + lx - 1])) {
                    lstart -= 1;
                    lx -= 1;
                    visited[4 + lx - sx][4 + dy] = 1;  // Mark as visited
                    digits[lstart] = d_grid[(y * n_cols) + lx];
                }

                // Gather digits from the right
                int rx = x;
                int rstart = start;
                while (rx <= n_cols && d_is_digit(d_grid[(y * n_cols) + rx + 1])) {
                    rstart += 1;
                    rx += 1;
                    visited[4 + rx - sx][4 + dy] = 1;  // Mark as visited
                    digits[rstart] = d_grid[(y * n_cols) + rx];
                }

                d_out[si] += parse_int(digits, lstart, 0, 0);
            }
        }
    }
}

__global__ void process_gear(int *d_grid, int n_cols, int n_rows, int *d_symbols, int n_symbols, int *d_out) {
    int si = blockIdx.x * blockDim.x + threadIdx.x;
    // Adjacency lookup [x offset, y offset]
    int adj[8][2] = {
        // Cardinals
        {0, 1},
        {0, -1},
        {1, 0},
        {-1, 0},
        // Diagonals
        {1, 1},
        {-1, -1},
        {1, -1},
        {-1, 1},
    };

    int visited[9][9];

    if (si < n_symbols) {
        int s = d_symbols[(si * 3) + 0];
        int sx = d_symbols[(si * 3) + 1];
        int sy = d_symbols[(si * 3) + 2];
        int ni = 0;
        int nums[2];
        
        for (int i = 0; i < 8; i++) {
            // Search all the adjacent positions
            int dx = adj[i][0];
            int dy = adj[i][1];
            int x = dx + sx;
            int y = dy + sy;

            if (x < 0 | x > n_cols) {
                continue;  // Outside of the grid
            }
            if (visited[4 + dx][4 + dy] == 1) {
                continue;  // Already visited this grid cell
            }
            visited[4 + dx][4 + dy] = 1;  // Mark as visited

            int value = d_grid[(y * n_cols) + x];
            if (d_is_digit(value)) {
                // Accumulate all the digits to the left and right
                int digits[9] = {
                    0, 0, 0,
                    0, 0, 0,
                    0, 0, 0,
                };
                int start = 4;
                digits[start] = value;

                // Gather digits from the left
                int lx = x;
                int lstart = start;
                while (lx >= 0 && d_is_digit(d_grid[(y * n_cols) + lx - 1])) {
                    lstart -= 1;
                    lx -= 1;
                    visited[4 + lx - sx][4 + dy] = 1;  // Mark as visited
                    digits[lstart] = d_grid[(y * n_cols) + lx];
                }

                // Gather digits from the right
                int rx = x;
                int rstart = start;
                while (rx <= n_cols && d_is_digit(d_grid[(y * n_cols) + rx + 1])) {
                    rstart += 1;
                    rx += 1;
                    visited[4 + rx - sx][4 + dy] = 1;  // Mark as visited
                    digits[rstart] = d_grid[(y * n_cols) + rx];
                }

                nums[ni] = parse_int(digits, lstart, 0, 0);
                ni += 1;
            }
        } 
        if (ni == 2 && s == 42) {
            d_out[si] = (nums[0] * nums[1]);
        }
    }
}

void part1(int *d_grid, int n_cols, int n_rows, int *d_symbols, int n_symbols) {
    int *d_symbol_totals = empty(n_symbols);
    process_symbol<<<blocks(n_symbols), threads(n_symbols)>>>(d_grid, n_cols, n_rows, d_symbols, n_symbols, d_symbol_totals);
    printf("part1: %d\n", from_device(sum(d_symbol_totals, n_symbols), 1)[0]);
}

void part2(int *d_grid, int n_cols, int n_rows, int *d_symbols, int n_symbols) {
    int *d_gear_totals = empty(n_symbols);
    process_gear<<<blocks(n_symbols), threads(n_symbols)>>>(d_grid, n_cols, n_rows, d_symbols, n_symbols, d_gear_totals);
    printf("part2: %d\n", from_device(sum(d_gear_totals, n_symbols), 1)[0]);
}

int main() {
    char *text = read_file((char*) "days/day03/input");

    int max_rows = 150;
    int max_cols = 150;
    int max_symbols = 10000;

    int grid[max_rows * max_cols];
    int symbols[max_symbols * 3];  // columns: symbol, x, y

    int row = 0;
    int col = 0;
    int symbol_row = 0;
    for (int i = 0; i < strlen(text); i++) {
        int value = text[i];
        if (value == 10) {
            // New line
            row += 1;
            col = 0;
            continue;
        }
        if (value == 46) {
            // Skip "."
            col += 1;
            continue;
        }
        if (is_digit(value)) {
            grid[(row * max_cols) + col] = value;
        } else {
            symbols[(symbol_row * 3) + 0] = value;
            symbols[(symbol_row * 3) + 1] = col;
            symbols[(symbol_row * 3) + 2] = row;
            symbol_row += 1;
        }
        col += 1;
    }
    int n_symbols = symbol_row;
    int *d_symbols = to_device(symbols, n_symbols * 3);
    int *d_grid = to_device(grid, max_rows * max_cols);

    part1(d_grid, max_cols, max_rows, d_symbols, n_symbols);
    part2(d_grid, max_cols, max_rows, d_symbols, n_symbols);

    return 0;
}
