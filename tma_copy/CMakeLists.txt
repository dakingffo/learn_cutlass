add_executable(tma_copy main.cu)

target_link_libraries(tma_copy 
    PRIVATE 
        nvidia::cutlass::cutlass
        CCCL::CCCL
)