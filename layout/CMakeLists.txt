add_executable(layout main.cpp tensor.cpp composition.cpp)

target_link_libraries(layout
    PRIVATE 
        nvidia::cutlass::cutlass
        CCCL::CCCL
        CUDA::cublas
)