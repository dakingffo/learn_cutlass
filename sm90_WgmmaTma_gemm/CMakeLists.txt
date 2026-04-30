add_executable(sm90_WgmmaTma_gemm main.cu)

target_link_libraries(sm90_WgmmaTma_gemm
    PRIVATE 
        nvidia::cutlass::cutlass
        CCCL::CCCL
        CUDA::cublas
)