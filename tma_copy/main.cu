#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/device_kernel.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/arch/mma_sm90.h>
#include <cutlass/util/helper_cuda.hpp>

#include "../utility/debug_step.cuh"
#include "../utility/timer.hpp"

#if defined(__CUDA_ARCH__)
  static_assert(__CUDA_ARCH__ >= 900);
#endif

namespace traits {
    using namespace cute;

    constexpr int TileM = 128;
    constexpr int TileN = 128;

    using type            = cute::half_t;
    using pointer         = type*;
    using const_pointer   = const type*;
    using SmemLayout      = decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<type>{}, Shape<Int<TileM>, Int<TileN>>{}));

    struct SharedStorage {
        alignas(128) ArrayEngine<type, cosize_v<SmemLayout>> S;
        uint64_t tma_barrier;
    };
}

template <typename TMALoad>
__global__ void copy_kernel(
    CUTLASS_GRID_CONSTANT const TMALoad tma_load,
    traits::const_pointer Sptr, traits::pointer Dptr,
    int m, int n
) {
    using namespace traits;
    __shared__ SharedStorage shared_storage;

    Tensor Shm = make_tensor(make_smem_ptr(shared_storage.S.begin()), SmemLayout{});
    Tensor mS  = tma_load.get_tma_tensor(make_shape(m, n));
    Tensor gS  = local_tile(mS, Tile<Int<TileM>, Int<TileN>>{}, make_coord(blockIdx.x, blockIdx.y));
    auto [tSgS, tSsS] = tma_partition(tma_load, Int<0>{}, Layout<_1>{}, group_modes<0,2>(Shm), group_modes<0,2>(gS));
    //Tensor D   = make_tensor(make_gmem_ptr(Dptr), make_shape(m, n), make_stride(n, Int<1>{}));

    int warp_idx              = cutlass::canonical_warp_idx_sync();
    int lane_predicate        = cute::elect_one_sync();
    int tma_transaction_bytes = sizeof(type) * TileM * TileN;
    using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;  // TMA

    if (warp_idx == 0 && lane_predicate) {
        ProducerBarType::init(&shared_storage.tma_barrier, 1);
    }
    cluster_sync();
    if (warp_idx == 0 && lane_predicate) {
        ProducerBarType::arrive_and_expect_tx(&shared_storage.tma_barrier, tma_transaction_bytes);
        copy(tma_load.with(shared_storage.tma_barrier), tSgS(_), tSsS(_));
    }
    ProducerBarType::wait(&shared_storage.tma_barrier, /*phase=*/0);
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

    Tensor mS = make_tensor(thrust::raw_pointer_cast(dS.data()), make_shape(m, n), make_stride(n, 1));
    SmemLayout sS;
    Copy_Atom tma_load = make_tma_atom(SM90_TMA_LOAD{}, mS, sS, Shape<Int<TileM>, Int<TileN>>{});

    dim3 block(16, 16);
    dim3 grid(ceil_div(m, TileM), ceil_div(n, TileN));

    std::cout << "Launching Copy kernel with grid: (" << grid.x << ", " << grid.y
              << "), block: " << block.x << std::endl;
    std::cout << "Tile sizes: M=" << TileM << ", N=" << TileN << std::endl;

    copy_kernel<<<grid, block>>>(
        tma_load,
        thrust::raw_pointer_cast(dS.data()), 
        thrust::raw_pointer_cast(dD.data()), 
        m, n
    );

    CUTE_CHECK_ERROR(cudaDeviceSynchronize());

    if (hD == dD) {
        std::cout << "SUCCESS: GPU results match CPU reference!" << std::endl;
    }
    else {
        std::cerr << "Failed!" << std::endl;
        return 0;
    }

    return 0;
}