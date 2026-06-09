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
    constexpr int Stage = 7;

    using type           = half_t; 
    using pointer        = type*; 
    using const_pointer  = const type*; 
    using mma_op         = SM90_64x64x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>;
    using mma_traits     = MMA_Traits<mma_op>;
    using mma_atom       = MMA_Atom<mma_traits>;
    using mma_atom_shape = typename mma_traits::Shape_MNK;

    constexpr int EURepeatM = 2;
    constexpr int EURepeatN = 1;
    constexpr int EURepeatK = 1;
    constexpr int PM = 1 * EURepeatM * get<0>(mma_atom_shape{});
    constexpr int PN = 1 * EURepeatN * get<1>(mma_atom_shape{});
    constexpr int PK = 1 * EURepeatK * get<2>(mma_atom_shape{});
    static_assert(TileM >= PM && TileN >= PN);

    using MMA = decltype(make_tiled_mma(
        mma_atom{},
        make_layout(Shape<Int<EURepeatM>, Int<EURepeatN>, Int<EURepeatK>>{}),
        Tile<Int<PM>, Int<PN>, Int<PK>>{}
    ));
    constexpr int ThreadsPerCTA = thr_size(MMA{});

    using r2s_copy_op      = SM90_U32x4_STSM_N;
    using r2s_copy_traits  = Copy_Traits<r2s_copy_op>;
    using r2s_copy_atom    = Copy_Atom<r2s_copy_traits, type>;
    using R2SCopyC         = decltype(make_tiled_copy_C(
        r2s_copy_atom{},
        MMA{}
    ));

    using SmemLayoutA = decltype(tile_to_shape(
        GMMA::Layout_K_SW128_Atom<type>{},
        Shape<Int<TileM>, Int<TileK>, Int<Stage>>{}
    ));
    using SmemLayoutB = decltype(tile_to_shape(
        GMMA::Layout_K_SW128_Atom<type>{},
        Shape<Int<TileN>, Int<TileK>, Int<Stage>>{}
    ));
    using SmemLayoutC = decltype(tile_to_shape(
        GMMA::Layout_K_SW128_Atom<type>{},
        Shape<Int<TileM>, Int<TileN>>{}
    ));
    constexpr int TMATransaction = sizeof(type) * (cosize(SmemLayoutA{}(_, _, 0)) + cosize(SmemLayoutB{}(_, _, 0)));
    static_assert(cosize_v<SmemLayoutA> + cosize_v<SmemLayoutB> >= cosize_v<SmemLayoutC>);

    struct SharedStorage {
        alignas(128)  ArrayEngine<type, cosize_v<SmemLayoutA>> A;
        alignas(128)  ArrayEngine<type, cosize_v<SmemLayoutB>> B;
        std::uint64_t tma_barrier[Stage];
    };
}

template <typename TMAStoreC, typename TMALoadA, typename TMALoadB>
__global__ __launch_bounds__(traits::ThreadsPerCTA) 
void gemm_kernel(
    CUTLASS_GRID_CONSTANT const TMAStoreC tma_store_c,
    CUTLASS_GRID_CONSTANT const TMALoadA  tma_load_a,
    CUTLASS_GRID_CONSTANT const TMALoadB  tma_load_b,
    int m, int n, int k
) {
#if defined(__CUDA_ARCH__)
    using namespace traits;

    extern __shared__ char shm_data[];
    auto& shared_storage = *reinterpret_cast<SharedStorage*>(shm_data);

    Tensor Acoord = tma_load_a.get_tma_tensor(make_shape(m, k));    // (M, K)
    Tensor Bcoord = tma_load_b.get_tma_tensor(make_shape(n, k));    // (N, K)
    Tensor Ccoord = tma_store_c.get_tma_tensor(make_shape(m, n));   // (M, N)

    Tensor gAcoord = local_tile(Acoord, make_tile(Int<TileM>{}, Int<TileK>{}), make_coord(blockIdx.x, _));   // (TileM, TileK, num_tile_K)
    Tensor gBcoord = local_tile(Bcoord, make_tile(Int<TileN>{}, Int<TileK>{}), make_coord(blockIdx.y, _));   // (TileN, TileK, num_tile_k)
    Tensor gCcoord = local_tile(Ccoord, make_tile(Int<TileM>{}, Int<TileN>{}), make_coord(blockIdx.x, blockIdx.y));  // (TileM, TileN)

    Tensor sA = make_tensor(make_smem_ptr(shared_storage.A.begin()), SmemLayoutA{});  // (TileM, TileK, Stage)
    Tensor sB = make_tensor(make_smem_ptr(shared_storage.B.begin()), SmemLayoutB{});  // (TileN, TileK, Stage)
    Tensor sC = make_tensor(make_smem_ptr(shared_storage.A.begin()), SmemLayoutC{});  // (TileM, TileN)

    // only used by one thread
    auto cta_tma_load_a = tma_load_a.get_slice(0);
    auto cta_tma_load_b = tma_load_b.get_slice(0);

    int global_idx  = 0;
    int shared_read = 0;
    int shared_recv = 0;

    if (threadIdx.x == 0) {
        for (auto& barrier : shared_storage.tma_barrier) {
            initialize_barrier(barrier, 1);
        }
        for (; shared_recv < Stage - 1; ++shared_recv, ++global_idx) {
            auto& recv_barrier = shared_storage.tma_barrier[shared_recv];
            set_barrier_transaction_bytes(recv_barrier, TMATransaction);
            copy(
                tma_load_a.with(recv_barrier),
                cta_tma_load_a.partition_S(gAcoord(_, _, global_idx)),
                cta_tma_load_a.partition_D(sA(_, _, shared_recv))
            );
            copy(
                tma_load_b.with(recv_barrier),
                cta_tma_load_b.partition_S(gBcoord(_, _, global_idx)),
                cta_tma_load_b.partition_D(sB(_, _, shared_recv))
            );
        }
    }
    __syncthreads();
    
    MMA tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tCrA = thr_mma.partition_fragment_A(sA);
    auto tCrB = thr_mma.partition_fragment_B(sB);
    auto tCrC = thr_mma.partition_fragment_C(sC);
    clear(tCrC);

    #pragma unroll 4
    for (int ik = 0; ik < k / TileK; ++ik, shared_read = (shared_read + 1) % Stage) {
        if (threadIdx.x == 0 && global_idx < k / TileK) {
            auto& recv_barrier = shared_storage.tma_barrier[shared_recv];
            set_barrier_transaction_bytes(recv_barrier, TMATransaction);
            copy(
                tma_load_a.with(recv_barrier),
                cta_tma_load_a.partition_S(gAcoord(_, _, global_idx)),
                cta_tma_load_a.partition_D(sA(_, _, shared_recv))
            );
            copy(
                tma_load_b.with(recv_barrier),
                cta_tma_load_b.partition_S(gBcoord(_, _, global_idx)),
                cta_tma_load_b.partition_D(sB(_, _, shared_recv))
            );
            global_idx++;
            shared_recv = (shared_recv + 1) % Stage;
        }
        auto& read_barrier = shared_storage.tma_barrier[shared_read];
        __syncthreads();
        wait_barrier(read_barrier, (ik / Stage) & 1);
        warpgroup_arrive();
        cute::gemm(tiled_mma, tCrA(_, _, _, shared_read), tCrB(_, _, _, shared_read), tCrC);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
    }
    __syncthreads();
    R2SCopyC r2s_copy_c;
    auto thr_r2s_copy_c = r2s_copy_c.get_slice(threadIdx.x);
    copy(r2s_copy_c, thr_r2s_copy_c.retile_S(tCrC), thr_r2s_copy_c.partition_D(sC));
    
    __syncthreads();
    tma_store_fence();
    if (threadIdx.x == 0) {
        auto cta_tma_store_c = tma_store_c.get_slice(0);
        copy(
            tma_store_c,
            cta_tma_store_c.partition_S(sC),
            cta_tma_store_c.partition_D(gCcoord)
        );
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
    Tensor gA = make_tensor(Aptr, make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor gB = make_tensor(Bptr, make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor gC = make_tensor(Cptr, make_shape(m, n), make_stride(n, Int<1>{}));
    auto sA = SmemLayoutA{};  // (TileM, TileK, Stage)
    auto sB = SmemLayoutB{};  // (TileN, TileK, Stage)
    auto sC = SmemLayoutC{};  // (TileM, TileN)

    // Create TMA Atoms with the desired copy operation on the source and destination
    auto tma_load_a = make_tma_copy(SM90_TMA_LOAD{}, gA, sA(_, _, 0));
    auto tma_load_b = make_tma_copy(SM90_TMA_LOAD{}, gB, sB(_, _, 0));
    auto tma_store_c = make_tma_copy(SM90_TMA_STORE{}, gC, sC(_, _));

    // Launch parameter setup
    int shm_size = sizeof(SharedStorage);
    dim3 block(ThreadsPerCTA);
    dim3 cluster(1, 1, 1);
    dim3 grid(
        round_up(ceil_div(m, TileM), cluster.x),
        round_up(ceil_div(n, TileN), cluster.y)
    );
    cutlass::ClusterLaunchParams params = {grid, block, cluster, shm_size};

    const void* kernel_ptr = reinterpret_cast<const void*>(
        &gemm_kernel<decltype(tma_store_c), decltype(tma_load_a), decltype(tma_load_b)>
    );

    CUTE_CHECK_ERROR(cudaFuncSetAttribute(
        kernel_ptr,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shm_size
    ));

    std::cout << "Launching GEMM kernel with grid: (" << grid.x << ", " << grid.y
              << "), block: " << block.x << std::endl;
    std::cout << "Tile sizes: M=" << TileM << ", N=" << TileN << ", K=" << TileK << std::endl;
    return std::make_tuple(params, kernel_ptr, tma_store_c, tma_load_a, tma_load_b);
}

int main() {
    using namespace traits;
    const int m = 4096;
    const int n = 4096;
    const int k = 4096;

    thrust::host_vector<type> hA(m * k);
    thrust::host_vector<type> hB(n * k);

    thrust::default_random_engine rng(123456);
    thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);

    thrust::generate(hA.begin(), hA.end(), [&]() { return type(dist(rng)); });
    thrust::generate(hB.begin(), hB.end(), [&]() { return type(dist(rng)); });

    thrust::device_vector<type> dA = hA;
    thrust::device_vector<type> dB = hB;
    thrust::device_vector<type> dC(m * n, type(0)), ref_dC(m * n, type(0));

    auto [params, kernel_ptr, tma_store_c, tma_load_a, tma_load_b] = setup_kernel(
        thrust::raw_pointer_cast(dC.data()), 
        thrust::raw_pointer_cast(dA.data()), 
        thrust::raw_pointer_cast(dB.data()), 
        m, n, k
    );

    cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
        tma_store_c, 
        tma_load_a, 
        tma_load_b,
        m, n, k
    );
    CUTE_CHECK_LAST();

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Error: Failed at kernel Launch" << std::endl;
    }

    CUTE_CHECK_ERROR(cudaDeviceSynchronize());

    std::cout << "Compare with reference result..." << std::endl;

    cublasHandle_t handle;
    cublasStatus_t custatus = cublasCreate(&handle);
    if (custatus != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS initialization failed!" << std::endl;
        return 1;
    }
    float alpha = 1.0f;
    float beta  = 0.0f;
    cublasGemmEx(
        handle, 
        CUBLAS_OP_T,   
        CUBLAS_OP_N,   
        n, m, k, 
        &alpha, 
        thrust::raw_pointer_cast(dB.data()), CUDA_R_16F, k,
        thrust::raw_pointer_cast(dA.data()), CUDA_R_16F, k,
        &beta, 
        thrust::raw_pointer_cast(ref_dC.data()), CUDA_R_16F, n,
        CUBLAS_COMPUTE_32F, 
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    CUTE_CHECK_ERROR(cudaDeviceSynchronize());

    thrust::host_vector<type> hC = dC, ref_hC = ref_dC;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float ref = static_cast<float>(ref_hC[i * n + j]);
            float res = static_cast<float>(hC[i * n + j]);
            if (std::fabs(ref - res) > 10) {
                std::cerr << "Expect " << ref << " at [" << i << "," << j << "], but get " << res << ".\n";
                return 0;
            }
        }
    }
    std::cout << "SUCCESS: GPU results match cuBLAS reference!" << std::endl;

    using namespace utility;
    TIMING("CUTE::GEMM", config<loop<5>>) {
        cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
            tma_store_c,
            tma_load_a, 
            tma_load_b,
            m, n, k
        );
        CUTE_CHECK_LAST();

        if (status != cutlass::Status::kSuccess) {
            std::cerr << "Error: Failed at kernel Launch" << std::endl;
        }
    }

    TIMING("CUBLAS::GEMM", config<loop<5>>) {
        cublasGemmEx(
            handle, 
            CUBLAS_OP_T,   
            CUBLAS_OP_N,   
            n, m, k, 
            &alpha, 
            thrust::raw_pointer_cast(dB.data()), CUDA_R_16F, k,
            thrust::raw_pointer_cast(dA.data()), CUDA_R_16F, k,
            &beta, 
            thrust::raw_pointer_cast(dC.data()), CUDA_R_16F, n,
            CUBLAS_COMPUTE_32F, 
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
    }
    cublasDestroy(handle);

    return 0;
}