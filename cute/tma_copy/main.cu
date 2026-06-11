#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>      // ClusterTransactionBarrier — CUTLASS TMA barrier wrapper
#include <cutlass/device_kernel.h>
#include <cutlass/util/helper_cuda.hpp>

#include <utility/timer.hpp>

#if defined(__CUDA_ARCH__)
  static_assert(__CUDA_ARCH__ >= 900);
#endif

namespace traits {
    using namespace cute;

    constexpr int TileM          = 128;
    constexpr int TileN          = 128;

    using type            = cute::half_t;
    using pointer         = type*;
    using const_pointer   = const type*;
    using SmemLayout      = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<type>{}, Shape<Int<TileM>, Int<TileN>>{}));

    constexpr int TMATransaction = sizeof(type) * cosize(SmemLayout{});

    struct SharedStorage {
        alignas(128) ArrayEngine<type, cosize_v<SmemLayout>> S;
        uint64_t tma_barrier;
    };

    using Barrier = cutlass::arch::ClusterTransactionBarrier;
}

template <typename TMALoad, typename TMAStore>
__global__ void copy_kernel(
    CUTLASS_GRID_CONSTANT const TMALoad tma_load,
    CUTLASS_GRID_CONSTANT const TMAStore tma_store,
    int m, int n
) {
    using namespace traits;
    __shared__ SharedStorage shared_storage;

    int warp_idx       = cutlass::canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();

    Tensor Shm      = make_tensor(make_smem_ptr(shared_storage.S.begin()), SmemLayout{});

    if (warp_idx == 0 && lane_predicate) {
        Tensor gScoord  = tma_load.get_tma_tensor(make_shape(m, n));
        Tensor cScoord  = local_tile(gScoord, Tile<Int<TileM>, Int<TileN>>{}, make_coord(blockIdx.x, blockIdx.y));
        auto cta_tma_load = tma_load.get_slice(0);
        Barrier::init(&shared_storage.tma_barrier, /*arrive_count=*/1);
        Barrier::arrive_and_expect_tx(&shared_storage.tma_barrier, TMATransaction);
        copy(
            tma_load.with(shared_storage.tma_barrier),
            cta_tma_load.partition_S(cScoord),
            cta_tma_load.partition_D(Shm)
        );
    }
    __syncthreads();
    Barrier::wait(&shared_storage.tma_barrier, /*phase=*/0);

    __syncthreads();
    tma_store_fence();
    if (warp_idx == 0 && lane_predicate) {
        Tensor gDcoord  = tma_store.get_tma_tensor(make_shape(m, n));
        Tensor cDcoord  = local_tile(gDcoord, Tile<Int<TileM>, Int<TileN>>{}, make_coord(blockIdx.x, blockIdx.y));
        auto cta_tma_store = tma_store.get_slice(0);
        copy(
            tma_store,
            cta_tma_store.partition_S(Shm),
            cta_tma_store.partition_D(cDcoord)
        );
    }
}

int main() {
    using namespace traits;
    const int m = 1024;
    const int n = 1024;

    thrust::host_vector<type> hS(m * n);
    thrust::default_random_engine rng(123456);
    thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
    thrust::generate(hS.begin(), hS.end(), [&]() { return type(dist(rng)); });
    thrust::host_vector<type> hD = hS;

    thrust::device_vector<type> dS = hS;
    thrust::device_vector<type> dD(m * n);

    Tensor gS = make_tensor(thrust::raw_pointer_cast(dS.data()), make_shape(m, n), make_stride(n, 1));
    auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, gS, SmemLayout{});
    Tensor gD = make_tensor(thrust::raw_pointer_cast(dD.data()), make_shape(m, n), make_stride(n, 1));
    auto tma_store = make_tma_copy(SM90_TMA_STORE{}, gD, SmemLayout{});

    dim3 block(256);
    dim3 grid(ceil_div(m, TileM), ceil_div(n, TileN));

    std::cout << "Launching Copy kernel with grid: (" << grid.x << ", " << grid.y
              << "), block: " << block.x << std::endl;
    std::cout << "Tile sizes: M=" << TileM << ", N=" << TileN << std::endl;

    copy_kernel<decltype(tma_load), decltype(tma_store)><<<grid, block>>>(
        tma_load, tma_store,
        m, n
    );

    CUTE_CHECK_ERROR(cudaDeviceSynchronize());

    if (hD == dD) {
        std::cout << "SUCCESS: GPU results match CPU reference!" << std::endl;
    }
    else {
        std::cerr << "Failed!" << std::endl;
        return 1;
    }

    return 0;
}
