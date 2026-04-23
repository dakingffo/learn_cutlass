#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/pipeline/pipeline.hpp>
#include <cublas_v2.h>

#include "../utility/timer.hpp"
#include "../utility/error_guard.hpp"

namespace traits {
    using namespace cute;
    using namespace cutlass;

    constexpr int TileM = 128;
    constexpr int TileN = 128;
    constexpr int TileK = 32;
    constexpr int Stage = 5;

    using type           = half_t; 
    using pointer        = type*; 
    using const_pointer  = const type*; 
    using mma_op         = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits     = MMA_Traits<mma_op>;
    using mma_atom       = MMA_Atom<mma_traits>;
    using mma_atom_shape = typename mma_traits::Shape_MNK;

    constexpr int ThreadsM = 2;
    constexpr int ThreadsN = 2;
    constexpr int ThreadsK = 1;
    constexpr int ExecuteM = 1 * ThreadsM * get<0>(mma_atom_shape{});
    constexpr int ExecuteN = 2 * ThreadsN * get<1>(mma_atom_shape{});
    constexpr int ExecuteK = 1 * ThreadsK * get<2>(mma_atom_shape{});

    using ThreadsArray = decltype(make_layout(
        Shape<Int<ThreadsM>, Int<ThreadsN>, Int<ThreadsK>>{}
    ));
    using ExecuteTasks = Tile<Int<ExecuteM>, Int<ExecuteN>, Int<ExecuteK>>;
    using MMA = decltype(make_tiled_mma(mma_atom{}, ThreadsArray{}, ExecuteTasks{}));

    using g2s_copy_op     = SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom   = Copy_Atom<g2s_copy_traits, type>;
    using G2SCopyA        = decltype(make_tiled_copy(
        g2s_copy_atom{},
        make_layout(Shape<_32, _4>{}, Stride<_4, _1>{}),
        make_layout(Shape<_1, _8>{})
    ));
    using G2SCopyB        = G2SCopyA;

    using s2r_copy_op     = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom   = Copy_Atom<s2r_copy_traits, type>;
    using S2RCopyA        = decltype(make_tiled_copy_A(
        s2r_copy_atom{},
        MMA{}
    ));
    using S2RCopyB        = decltype(make_tiled_copy_B(
        s2r_copy_atom{},
        MMA{}
    ));

    using R2SCopyAtomC    = Copy_Atom<UniversalCopy<int>, type>;
    using R2SCopyC        = decltype(make_tiled_copy_C(
        R2SCopyAtomC{},
        MMA{}
    ));
    using S2GCopyAtomC    = Copy_Atom<UniversalCopy<uint128_t>, type>;
    using S2GCopyC        = decltype(make_tiled_copy(
        S2GCopyAtomC{},
        make_layout(Shape<_32, _4>{}, Stride<_4, _1>{}),
        make_layout(Shape<_1, _8>{})
    ));

    using SmemLayoutA = decltype(tile_to_shape(
        composition(
            Swizzle<2, 3, 3>{},
            make_layout(Shape<_8, Int<TileK>>{}, Stride<Int<TileK>, _1>{})
        ),
        Shape<Int<TileM>, Int<TileK>, Int<Stage>>{}
    ));
    using SmemLayoutB = decltype(tile_to_shape(
        composition(
            Swizzle<2, 3, 3>{},
            make_layout(Shape<_8, Int<TileK>>{}, Stride<Int<TileK>, _1>{})
        ),
        Shape<Int<TileN>, Int<TileK>, Int<Stage>>{}
    ));
    static_assert((TileM + TileN) * (TileK + Stage) >= ExecuteM * ExecuteN * 2);
    using SmemLayoutC = decltype(tile_to_shape(
        composition(
            Swizzle<2, 3, 3>{}, 
            make_layout(Shape<Int<ExecuteM>, Int<ExecuteN>>{}, Stride<Int<ExecuteN>, _1>{})
        ),
        Shape<Int<ExecuteM>, Int<ExecuteN>, _2>{}
    ));

    using Pipe       = PipelineAsync<Stage>;
    using PipeState  = PipelineState<Stage>;
    using PipeParams = typename Pipe::Params;
    struct SharedStorage {
        half shm_[cosize(SmemLayoutA{}) + cosize(SmemLayoutB{})];
        typename Pipe::SharedStorage pipe_shared_storage_;
    };
}

__global__ void gemm_kernel(
    traits::pointer       Cptr, 
    traits::const_pointer Aptr, 
    traits::const_pointer Bptr, 
    int m, int n, int k
) {
    using namespace traits;
    extern __shared__ type shm_data[];

    auto& shared_storage = *reinterpret_cast<SharedStorage*>(shm_data);
    Pipe pipe(shared_storage.pipe_shared_storage_, 
              PipeParams{.producer_arv_count = blockDim.x * blockDim.y, .consumer_arv_count = blockDim.x * blockDim.y}
    );
    PipeState smem_pipe_write = make_producer_start_state<Pipe>();
    PipeState smem_pipe_read;

    pointer Ashm = shm_data;
    pointer Bshm = shm_data + cosize(SmemLayoutA{});

    int idx = threadIdx.x;
    int ix  = blockIdx.x;
    int iy  = blockIdx.y;

    Tensor A = make_tensor(make_gmem_ptr((const_pointer)Aptr), make_shape(m, k), make_stride(k, Int<1>{}));  // (M, K)
    Tensor B = make_tensor(make_gmem_ptr((const_pointer)Bptr), make_shape(n, k), make_stride(k, Int<1>{}));  // (N, K)
    Tensor C = make_tensor(make_gmem_ptr((pointer)Cptr), make_shape(m, n), make_stride(n, Int<1>{}));        // (M, N)

    Tensor gA = local_tile(A, make_tile(Int<TileM>{}, Int<TileK>{}), make_coord(iy, _));   // (TileM, TileK, num_tile_K)
    Tensor gB = local_tile(B, make_tile(Int<TileN>{}, Int<TileK>{}), make_coord(ix, _));   // (TileN, TileK, num_tile_k)
    Tensor gC = local_tile(C, make_tile(Int<TileM>{}, Int<TileN>{}), make_coord(iy, ix));  // (TileM, TileN)

    Tensor sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});  // (TileM, TileK, Stage)
    Tensor sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});  // (TileN, TileK, Stage)

    MMA tiled_mma;
    auto this_mma = tiled_mma.get_slice(idx);
    auto rA       = this_mma.partition_fragment_A(gA(_, _, 0));  // (num_mma, MMA_M, MMA_K)
    auto rB       = this_mma.partition_fragment_B(gB(_, _, 0));  // (num_mma, MMA_N, MMA_K)
    auto rC       = this_mma.partition_fragment_C(gC);           // (num_mma, MMA_M, MMA_N)
    clear(rC);

    // A: gmem -cp.async-> shm -ldmatrix-> reg
    G2SCopyA g2s_copy_a;
    auto this_g2s_copy_a = g2s_copy_a.get_slice(idx);
    auto g2sA_src        = this_g2s_copy_a.partition_S(gA);  // (num_copy, CPY_M, CPY_K, k)
    auto g2sA_dst        = this_g2s_copy_a.partition_D(sA);  // (num_copy, CPY_M, CPY_K, Stage)

    S2RCopyA s2r_copy_a;
    auto this_s2r_copy_a = s2r_copy_a.get_slice(idx);
    auto s2rA_src        = this_s2r_copy_a.partition_S(sA);  // (num_copy, CPY_M, CPY_K, Stage)
    auto s2rA_dst        = this_s2r_copy_a.retile_D(rA);     // (num_copy, CPY_M, CPY_K)

    // B: gmem -cp.async-> shm -ldmatrix-> reg
    G2SCopyB g2s_copy_b;
    auto this_g2s_copy_b = g2s_copy_b.get_slice(idx);
    auto g2sB_src        = this_g2s_copy_b.partition_S(gB); // (num_copy, CPY_N, CPY_K, k)
    auto g2sB_dst        = this_g2s_copy_b.partition_D(sB);  // (num_copy, CPY_N, CPY_K, Stage)

    S2RCopyB s2r_copy_b;
    auto this_s2r_copy_b = s2r_copy_b.get_slice(idx);
    auto s2rB_src        = this_s2r_copy_b.partition_S(sB);  // (num_copy, CPY_N, CPY_K, Stage)
    auto s2rB_dst        = this_s2r_copy_b.retile_D(rB);     // (num_copy, CPY_N, CPY_K)

    int global_idx  = 0;
    // gmem -> shm
    #pragma unroll
    for (; global_idx < Stage - 1; ++global_idx, ++smem_pipe_write) {
        pipe.producer_acquire(smem_pipe_write);
        copy(g2s_copy_a, g2sA_src(_, _, _, global_idx), g2sA_dst(_, _, _, smem_pipe_write.index()));
        copy(g2s_copy_b, g2sB_src(_, _, _, global_idx), g2sB_dst(_, _, _, smem_pipe_write.index()));
        pipe.producer_commit(smem_pipe_write);
    }
    // smem -> reg
    pipe.consumer_wait(smem_pipe_read);
    copy(s2r_copy_a, s2rA_src(_, _, 0, smem_pipe_read.index()), s2rA_dst(_, _, 0));
    copy(s2r_copy_b, s2rB_src(_, _, 0, smem_pipe_read.index()), s2rB_dst(_, _, 0));
    ++smem_pipe_read;

    #pragma unroll 4
    for (int it = 0; it < k / TileK; ++it) {
        #pragma unroll
        for (int ik = 0; ik < size<2>(rA); ++ik) {
            if (ik == 0) {
                if (global_idx < k / TileK) {
                    pipe.producer_acquire(smem_pipe_write);
                    copy(g2s_copy_a, g2sA_src(_, _, _, global_idx), g2sA_dst(_, _, _, smem_pipe_write.index()));
                    copy(g2s_copy_b, g2sB_src(_, _, _, global_idx), g2sB_dst(_, _, _, smem_pipe_write.index()));
                    ++smem_pipe_write, ++global_idx;
                }                
                pipe.producer_commit(smem_pipe_write);
            }
            if (ik == size<2>(rA) - 1) {
                pipe.consumer_release(smem_pipe_read);
                ++smem_pipe_read;
                pipe.consumer_wait(smem_pipe_read);
            }
            int ik_next = (ik + 1) % size<2>(rA);
            copy(s2r_copy_a, s2rA_src(_, _, ik_next, smem_pipe_read.index()), s2rA_dst(_, _, ik_next));
            copy(s2r_copy_b, s2rB_src(_, _, ik_next, smem_pipe_read.index()), s2rB_dst(_, _, ik_next));
            gemm(tiled_mma, rC, rA(_, _, ik), rB(_, _, ik), rC);
        }
    }  

    // C: reg -> shm -> gmem
    Tensor sC = make_tensor(make_smem_ptr(shm_data), SmemLayoutC{});

    R2SCopyC r2s_copy_c;
    auto this_r2s_copy_c = r2s_copy_c.get_slice(idx);
    auto r2sC_srcx       = group_modes<1, 3>(this_r2s_copy_c.retile_S(rC));     // (num_copy, CPY_M, CPY_N) -> (num_copy, CPY_MN)
    auto r2sC_dst        = this_r2s_copy_c.partition_D(sC);                     // (num_copy, 1, 1, pipe)
    
    S2GCopyC s2g_copy_c;
    auto this_s2g_copy_c = s2g_copy_c.get_thread_slice(idx);
    auto s2gC_src        = this_s2g_copy_c.partition_S(sC);                     // (num_copy, 1, 1, pipe)
    auto s2gC_dstx       = group_modes<1, 3>(this_s2g_copy_c.partition_D(gC));  // (num_copy, CPY_MN) -> (num_copy, CPY_M, CPY_N)

    #pragma unroll
    for (int i = 0, step = size<3>(r2sC_dst); i < size<1>(r2sC_srcx); i += step) {
        #pragma unroll
        for (int j = 0; j < step; ++j) /* reg -> shm */ {
            copy(r2s_copy_c, r2sC_srcx(_, i + j), r2sC_dst(_, 0, 0, j));
        }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < step; ++j) /* shm -> gmem */ {
            copy(s2g_copy_c, s2gC_src(_, 0, 0, j), s2gC_dstx(_, i + j));
        }
        __syncthreads();
    }
}

int main() {
    using namespace traits;
    const int m = 2048;
    const int n = 2048;
    const int k = 2048;

    thrust::host_vector<type> hA(m * k);
    thrust::host_vector<type> hB(n * k);

    thrust::default_random_engine rng(123456);
    thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);

    thrust::generate(hA.begin(), hA.end(), [&]() { return type(dist(rng)); });
    thrust::generate(hB.begin(), hB.end(), [&]() { return type(dist(rng)); });

    thrust::device_vector<type> dA = hA;
    thrust::device_vector<type> dB = hB;
    thrust::device_vector<type> dC(m * n, type(0));

    dim3 block(cute::size(MMA{}));
    dim3 grid((n + TileN - 1) / TileN, (m + TileM - 1) / TileM);

    std::cout << "Launching GEMM kernel with grid: (" << grid.x << ", " << grid.y
              << "), block: " << block.x << std::endl;
    std::cout << "Tile sizes: M=" << TileM << ", N=" << TileN << ", K=" << TileK << std::endl;

    int shm_size = sizeof(SharedStorage);
    ERROR_GUARD(cudaFuncSetAttribute(
        gemm_kernel, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, 
        shm_size
    ));

    gemm_kernel<<<grid, block, shm_size>>>(
        thrust::raw_pointer_cast(dC.data()), 
        thrust::raw_pointer_cast(dA.data()), 
        thrust::raw_pointer_cast(dB.data()), 
        m, n, k
    );

    ERROR_GUARD(cudaDeviceSynchronize());

    thrust::host_vector<type> hC = dC;

    std::cout << "Computing reference CPU result..." << std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int t = 0; t < k; ++t) {
                float a = static_cast<float>(hA[i * k + t]);
                float b = static_cast<float>(hB[j * k + t]);
                sum += a * b;
            }
            float expected = static_cast<float>(hC[i * n + j]);
            if (std::abs(sum - expected) > 10) {
                std::cerr << "Expect " << expected << " at [" << i << "," << j << "], but get " << sum << ".\n";
                return 0;
            }
        }
    }
    std::cout << "SUCCESS: GPU results match CPU reference!" << std::endl;

    using namespace utility;
    TIMING("CUTE::GEMM", config<loop<5>>) {
        gemm_kernel<<<grid, block, shm_size>>>(
            thrust::raw_pointer_cast(dC.data()), 
            thrust::raw_pointer_cast(dA.data()), 
            thrust::raw_pointer_cast(dB.data()), 
            m, n, k
        );
    }

    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS initialization failed!" << std::endl;
        return 1;
    }
    float alpha = 1.0f;
    float beta  = 0.0f;
    TIMING("CUBLAS::GEMM", config<loop<5>>) {
        cublasGemmEx(
            handle, 
            CUBLAS_OP_T,   
            CUBLAS_OP_N,   
            m, n, k, 
            &alpha, 
            thrust::raw_pointer_cast(dA.data()), CUDA_R_16F, k,
            thrust::raw_pointer_cast(dB.data()), CUDA_R_16F, k,
            &beta, 
            thrust::raw_pointer_cast(dC.data()), CUDA_R_16F, m,
            CUBLAS_COMPUTE_32F, 
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
    }
    cublasDestroy(handle);

    return 0;
}