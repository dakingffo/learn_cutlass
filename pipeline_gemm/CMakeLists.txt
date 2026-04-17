add_executable(pipeline_gemm main.cu)

target_link_libraries(pipeline_gemm 
    PRIVATE 
        nvidia::cutlass::cutlass
        CCCL::CCCL
        CUDA::cublas
)