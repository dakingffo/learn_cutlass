cmake_minimum_required(VERSION 3.20)
project(learn_cutlass LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(NvidiaCutlass REQUIRED)
find_package(CCCL REQUIRED)

add_library(Utility INTERFACE)

target_include_directories(Utility
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(Utility
    INTERFACE
        nvidia::cutlass::cutlass
        CCCL::CCCL           
        CUDA::cublas
)

add_subdirectory(cute)