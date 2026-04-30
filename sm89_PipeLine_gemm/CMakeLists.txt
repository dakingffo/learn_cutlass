add_executable(sm89_PipeLine_gemm main.cu)

target_link_libraries(sm89_PipeLine_gemm
    PRIVATE 
        nvidia::cutlass::cutlass
        CCCL::CCCL
        CUDA::cublas
)