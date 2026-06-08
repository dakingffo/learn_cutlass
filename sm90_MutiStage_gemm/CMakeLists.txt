add_executable(sm90_MutiStage_gemm main.cu)

target_link_libraries(sm90_MutiStage_gemm
    PRIVATE 
        nvidia::cutlass::cutlass
        CCCL::CCCL
        CUDA::cublas
)