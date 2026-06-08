add_executable(sm90_SPMCPipe_gemm main.cu)

target_link_libraries(sm90_SPMCPipe_gemm
    PRIVATE 
        nvidia::cutlass::cutlass
        CCCL::CCCL
        CUDA::cublas
)