add_executable(simple_copy main.cu)

target_link_libraries(simple_copy 
    PRIVATE 
        nvidia::cutlass::cutlass
        CCCL::CCCL
)