#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/util/print_error.hpp>
#include <cutlass/util/helper_cuda.hpp>
#include <cutlass/arch/mma_sm90.h>
#include <cutlass/device_kernel.h>
#include <cublas_v2.h>

#include "../utility/timer.hpp"

#if defined(__CUDA_ARCH__)
  static_assert(__CUDA_ARCH__ >= 900);
#endif

namespace traits {
    using namespace cute;
    using namespace cutlass;

    constexpr int TileM = 128;
    constexpr int TileN = 128;
    constexpr int TileK = 64;
    constexpr int Stage = 5;

    using type           = half_t; 
    using pointer        = type*; 
    using const_pointer  = const type*; 
    using mma_op         = SM90_64x64x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>;
    using mma_traits     = MMA_Traits<mma_op>;
    using mma_atom       = MMA_Atom<mma_traits>;
    using mma_atom_shape = typename mma_traits::Shape_MNK;

    constexpr int ThreadsM = 1;
    constexpr int ThreadsN = 1;
    constexpr int ThreadsK = 2;
    constexpr int ExecuteM = 1 * ThreadsM * get<0>(mma_atom_shape{});
    constexpr int ExecuteN = 1 * ThreadsN * get<1>(mma_atom_shape{});
    constexpr int ExecuteK = 1 * ThreadsK * get<2>(mma_atom_shape{});

    using ThreadsArray = decltype(make_layout(
        Shape<Int<ThreadsM>, Int<ThreadsN>, Int<ThreadsK>>{}
    ));
    using ExecuteTasks = Tile<Int<ExecuteM>, Int<ExecuteN>, Int<ExecuteK>>;
    // using MMA = decltype(make_tiled_mma(mma_atom{}, ThreadsArray{}, ExecuteTasks{}));
    using MMA = decltype(make_tiled_mma(mma_atom{}));

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
        GMMA::Layout_K_SW128_Atom<type>{},
        Shape<Int<TileM>, Int<TileK>, Int<Stage>>{}
    ));
    using SmemLayoutB = decltype(tile_to_shape(
        GMMA::Layout_K_SW128_Atom<type>{},
        Shape<Int<TileN>, Int<TileK>, Int<Stage>>{}
    ));
    static_assert((TileM + TileN) * (TileK + Stage) >= ExecuteM * ExecuteN * 2);
    using SmemLayoutC = decltype(tile_to_shape(
        GMMA::Layout_K_SW128_Atom<type>{},
        Shape<Int<ExecuteM>, Int<ExecuteN>, _2>{}
    ));

    struct SharedStorage {
        alignas(128)  ArrayEngine<type, cosize_v<SmemLayoutA>> A;
        alignas(128)  ArrayEngine<type, cosize_v<SmemLayoutB>> B;
        std::uint64_t tma_barrier[size<2>(SmemLayoutA{})];
        std::uint64_t mma_barrier[size<2>(SmemLayoutA{})];
    };
}

template <typename TMALoadA, typename TMALoadB>
__global__ __launch_bounds__(decltype(cute::size(traits::MMA{}))::value) 
void gemm_kernel(
    traits::pointer       Cptr, 
    traits::const_pointer Aptr, 
    traits::const_pointer Bptr, 
    CUTLASS_GRID_CONSTANT const TMALoadA tma_load_a,
    CUTLASS_GRID_CONSTANT const TMALoadB tma_load_b,
    int m, int n, int k
) {
#if defined(__CUDA_ARCH__)
    using namespace traits;
    extern __shared__ char shm_data[];
    auto& shared_storage = *reinterpret_cast<SharedStorage*>(shm_data);

    int idx = threadIdx.x;
    int ix  = blockIdx.x;
    int iy  = blockIdx.y;

    Tensor mA = tma_load_a.get_tma_tensor(make_shape(m, k));    // (M, K)
    Tensor mB = tma_load_b.get_tma_tensor(make_shape(n, k));    // (N, K)
    Tensor mC = make_tensor(make_gmem_ptr((pointer)Cptr), make_shape(m, n), make_stride(n, Int<1>{}));  // (M, N)

    Tensor gA = local_tile(mA, make_tile(Int<TileM>{}, Int<TileK>{}), make_coord(iy, _));   // (TileM, TileK, num_tile_K)
    Tensor gB = local_tile(mB, make_tile(Int<TileN>{}, Int<TileK>{}), make_coord(ix, _));   // (TileN, TileK, num_tile_k)
    Tensor gC = local_tile(mC, make_tile(Int<TileM>{}, Int<TileN>{}), make_coord(iy, ix));  // (TileM, TileN)

    Tensor sA = make_tensor(make_smem_ptr(shared_storage.A.begin()), SmemLayoutA{});  // (TileM, TileK, Stage)
    Tensor sB = make_tensor(make_smem_ptr(shared_storage.B.begin()), SmemLayoutB{});  // (TileN, TileK, Stage)

    auto [tAgA, tAsA] = tma_partition(tma_load_a, Int<0>{}, Layout<_1>{}, group_modes<0,2>(sA), group_modes<0,2>(gA));
    // (TMA, k) and (TMA, Stage)
    auto [tBgB, tBsB] = tma_partition(tma_load_b, Int<0>{}, Layout<_1>{}, group_modes<0,2>(sB), group_modes<0,2>(gB));
    // (TMA, k) and (TMA, Stage) 
    constexpr int tma_transaction_bytes = sizeof(make_tensor_like(tensor<0>(tAsA))) + sizeof(make_tensor_like(tensor<0>(tBsB)));

    int ik = 0;
    int k_tile_count = size<1>(tAgA);

    // Initialize Barriers
    int warp_idx  = cutlass::canonical_warp_idx_sync();
    int is_leader = cute::elect_one_sync();
    uint64_t* producer_mbar = shared_storage.tma_barrier;
    uint64_t* consumer_mbar = shared_storage.mma_barrier;

    using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;  // TMA
    using ConsumerBarType = cutlass::arch::ClusterBarrier;             // MMA

    CUTE_UNROLL
    for (int pipe = 0; pipe < Stage; ++pipe) {
        if ((warp_idx == 0) && is_leader) {
            ProducerBarType::init(&producer_mbar[pipe], 1);
            ConsumerBarType::init(&consumer_mbar[pipe], 128);
        }
    }

    cluster_sync();
    CUTE_UNROLL
    for (int pipe = 0; pipe < Stage; ++pipe){
        if ((warp_idx == 0) && is_leader)
        {
            ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
            copy(tma_load_a.with(producer_mbar[pipe]), tAgA(_,ik), tAsA(_,pipe));
            copy(tma_load_b.with(producer_mbar[pipe]), tBgB(_,ik), tBsB(_,pipe));
        }
        --k_tile_count;
        ++ik;
    }

    MMA mma;
    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

    Tensor tCrA = thr_mma.make_fragment_A(tCsA);                         // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);                         // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)
    clear(tCrC);

    auto smem_pipe_write = cutlass::PipelineState<Stage>();             
    auto smem_pipe_read  = cutlass::PipelineState<Stage>();   

    CUTE_NO_UNROLL
    while (k_tile_count > -Stage)
    {
        int pipe = smem_pipe_write.index();
        ProducerBarType::wait(&producer_mbar[pipe], smem_pipe_write.phase());

        // MMAs to cover 1 K_TILE
        warpgroup_arrive();
        cute::gemm(mma, tCrA(_,_,_,pipe), tCrB(_,_,_,pipe), tCrC);     // (V,M) x (V,N) => (V,M,N)
        warpgroup_commit_batch();
        warpgroup_wait<0>();

        ConsumerBarType::arrive(&consumer_mbar[pipe]);
        ++smem_pipe_read;

        // Only issue new TMA copies if there are more tiles to fetch
        if ((warp_idx == 0) && is_leader && (k_tile_count > 0)) {
            pipe = smem_pipe_write.index();
            ConsumerBarType::wait(&consumer_mbar[pipe], smem_pipe_write.phase());
            ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
            copy(tma_load_a.with(producer_mbar[pipe]), tAgA(_,ik), tAsA(_,pipe));
            copy(tma_load_b.with(producer_mbar[pipe]), tBgB(_,ik), tBsB(_,pipe));
            ++smem_pipe_write;
        }
        --k_tile_count;
        ++ik;
    }
    
    // C: reg -> shm -> gmem
    Tensor sC = make_tensor(make_smem_ptr((type*)shm_data), SmemLayoutC{});

    R2SCopyC r2s_copy_c;
    auto this_r2s_copy_c = r2s_copy_c.get_slice(idx);
    auto r2sC_srcx       = group_modes<1, 3>(this_r2s_copy_c.retile_S(tCrC));     // (num_copy, CPY_M, CPY_N) -> (num_copy, CPY_MN)
    auto r2sC_dst        = this_r2s_copy_c.partition_D(sC);                     // (num_copy, 1, 1, pipe)
    
    S2GCopyC s2g_copy_c;
    auto this_s2g_copy_c = s2g_copy_c.get_thread_slice(idx);
    auto s2gC_src        = this_s2g_copy_c.partition_S(sC);                     // (num_copy, 1, 1, pipe)
    auto s2gC_dstx       = group_modes<1, 3>(this_s2g_copy_c.partition_D(gC));  // (num_copy, CPY_MN) -> (num_copy, CPY_M, CPY_N)

    CUTE_NO_UNROLL
    for (int i = 0, step = size<3>(r2sC_dst); i < size<1>(r2sC_srcx); i += step) {
        CUTE_NO_UNROLL
        for (int j = 0; j < step; ++j) /* reg -> shm */ {
            copy(r2s_copy_c, r2sC_srcx(_, i + j), r2sC_dst(_, 0, 0, j));
        }
        __syncthreads();
        CUTE_NO_UNROLL
        for (int j = 0; j < step; ++j) /* shm -> gmem */ {
            copy(s2g_copy_c, s2gC_src(_, 0, 0, j), s2gC_dstx(_, i + j));
        }
        __syncthreads();
    }
#endif 
}

auto setup_kernel(    
    traits::pointer       Cptr, 
    traits::const_pointer Aptr, 
    traits::const_pointer Bptr, 
    int m, int n, int k
) {
    using namespace traits;
    Tensor mA = make_tensor(Aptr, make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor mB = make_tensor(Bptr, make_shape(n, k), make_stride(k, Int<1>{}));

    auto sA = SmemLayoutA{};  // (TileM, TileK, Stage)
    auto sB = SmemLayoutB{};  // (TileN, TileK

    // Create TMA Atoms with the desired copy operation on the source and destination
    Copy_Atom tma_load_a = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_, _, 0), Shape<Int<TileM>, Int<TileK>>{});
    Copy_Atom tma_load_b = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_, _, 0), Shape<Int<TileN>, Int<TileK>>{});

    // Launch parameter setup
    int shm_size = int(sizeof(SharedStorage));
    dim3 dimBlock(size(MMA{}));
    dim3 dimCluster(2, 1, 1);
    dim3 dimGrid(
        round_up(size(ceil_div(m, TileM)), dimCluster.x),
        round_up(size(ceil_div(n, TileN)), dimCluster.y)
    );
    cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, shm_size};

    const void* kernel_ptr = reinterpret_cast<const void*>(&gemm_kernel<decltype(tma_load_a),decltype(tma_load_b)>);

    CUTE_CHECK_ERROR(cudaFuncSetAttribute(
        kernel_ptr,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shm_size
    ));

    std::cout << "Launching GEMM kernel with grid: (" << dimGrid.x << ", " << dimGrid.y
              << "), block: " << dimBlock.x << std::endl;
    std::cout << "Tile sizes: M=" << TileM << ", N=" << TileN << ", K=" << TileK << std::endl;

    return make_tuple(params, kernel_ptr, tma_load_a, tma_load_b);
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

    auto [params, kernel_ptr, tma_load_a, tma_load_b] = setup_kernel(
        thrust::raw_pointer_cast(dC.data()), 
        thrust::raw_pointer_cast(dA.data()), 
        thrust::raw_pointer_cast(dB.data()), 
        m, n, k
    );

    // Kernel Launch
    cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
        thrust::raw_pointer_cast(dC.data()), 
        thrust::raw_pointer_cast(dA.data()), 
        thrust::raw_pointer_cast(dB.data()), 
        tma_load_a, tma_load_b,
        m, n, k
    );
    CUTE_CHECK_LAST();

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Error: Failed at kernel Launch" << std::endl;
    }

    CUTE_CHECK_ERROR(cudaDeviceSynchronize());

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
        cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
            thrust::raw_pointer_cast(dC.data()), 
            thrust::raw_pointer_cast(dA.data()), 
            thrust::raw_pointer_cast(dB.data()), 
            tma_load_a, tma_load_b,
            m, n, k
        );
        CUTE_CHECK_LAST();

        if (status != cutlass::Status::kSuccess) {
            std::cerr << "Error: Failed at kernel Launch" << std::endl;
        }
    }

    cublasHandle_t handle;
    cublasStatus_t custatus = cublasCreate(&handle);
    if (custatus != CUBLAS_STATUS_SUCCESS) {
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