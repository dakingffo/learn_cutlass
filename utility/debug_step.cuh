#pragma once

#include <cstdio>

#define DEBUG_STEP(step_idx) \
    __syncthreads(); \
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) { \
        printf("Reached Step %d\n", step_idx); \
    } \
    __syncthreads();