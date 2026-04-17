add_executable(simple_gemm main.cu)

target_link_libraries(simple_gemm 
    PRIVATE 
        nvidia::cutlass::cutlass
        CCCL::CCCL
)