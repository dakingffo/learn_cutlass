add_executable(example_wgmma_tma_sm90 wgmma_tma_sm90.cu)

target_link_libraries(example_wgmma_tma_sm90
    PRIVATE 
        nvidia::cutlass::cutlass
        CCCL::CCCL
)