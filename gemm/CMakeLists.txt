add_executable(gemm main.cu)

target_link_libraries(gemm 
    PRIVATE 
        nvidia::cutlass::cutlass
        CCCL::CCCL
        CUDA::cublas
)