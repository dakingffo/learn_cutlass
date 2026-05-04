cmake_minimum_required(VERSION 3.20)
project(learn_cutlass LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(NvidiaCutlass REQUIRED)
find_package(CCCL REQUIRED)

add_subdirectory(cutlass_example)
add_subdirectory(layout)
add_subdirectory(simple_copy)
add_subdirectory(tma_copy)
add_subdirectory(simple_gemm)
add_subdirectory(sm89_PipeLine_gemm)
add_subdirectory(sm90_WgmmaTma_gemm)
add_subdirectory(sm90_SPMCPipe_gemm)