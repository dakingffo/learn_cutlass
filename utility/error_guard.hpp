#pragma once

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cstdio>

#define ERROR_GUARD(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)