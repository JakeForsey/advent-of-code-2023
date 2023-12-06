#include <stdio.h>

#include "../../lib/aoc.h"
#include "../../lib/cuda.h"
#include "../../lib/ops.h"

__global__ void run(int *d_times, long *d_distances, int i,  int *d_out) {
    int charge_millis = blockIdx.x * blockDim.x + threadIdx.x;
    if (charge_millis < d_times[i]) {
        int remaining = d_times[i] - charge_millis;
        long distance = (long) charge_millis * (long) remaining;
        if ((long) distance > (long) d_distances[i]) {
            d_out[charge_millis] += 1;
        }
    }
}

int dispatch(int *d_times, long *d_distances, int n) {
    int result = 1;
    for (int i = 0; i < n; i++) {
        int millis = from_device(d_times, n)[i];
        int *d_out = empty(millis);
        run<<<blocks(millis), threads(millis)>>>(d_times, d_distances, i, d_out);
        result *= from_device(sum(d_out, millis), 1)[0];
    }
    return result;
}

void part1(int *d_times, long *d_distances, int n) {
    printf("part1: %d\n", dispatch(d_times, d_distances, n));
}

void part2(int *d_times, long *d_distances, int n) {
    printf("part2: %d\n", dispatch(d_times, d_distances, n));
}

int main() {
    int p1_times[4] = {53, 71, 78, 80};
    long p1_distances[4] = {275, 1181, 1215, 1524};

    int p2_time[1] = {53717880};
    long p2_distance[1] = {275118112151524};

    part1(to_device(p1_times, 4), to_device_long(p1_distances, 4), 4);
    part2(to_device(p2_time, 1), to_device_long(p2_distance, 1), 1);

    return 0;
}
