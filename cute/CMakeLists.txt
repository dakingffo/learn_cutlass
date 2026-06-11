add_executable(simple_copy simple_copy/main.cu)
target_link_libraries(simple_copy Utility)

add_executable(tma_copy tma_copy/main.cu)
target_link_libraries(tma_copy Utility)

add_executable(simple_gemm simple_gemm/main.cu)
target_link_libraries(simple_gemm Utility)

add_executable(sm80_MutiStage_gemm sm80_MutiStage_gemm/main.cu)
target_link_libraries(sm80_MutiStage_gemm Utility)

add_executable(sm90_MutiStage_gemm sm90_MutiStage_gemm/main.cu)
target_link_libraries(sm90_MutiStage_gemm Utility)

add_executable(sm90_WS_gemm sm90_WS_gemm/main.cu)
target_link_libraries(sm90_WS_gemm Utility)