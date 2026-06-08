add_executable(sm80_MutiStage_gemm main.cu)

target_link_libraries(sm80_MutiStage_gemm
    PRIVATE 
        nvidia::cutlass::cutlass
        CCCL::CCCL
        CUDA::cublas
)