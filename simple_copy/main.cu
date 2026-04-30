#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

#include "../utility/timer.hpp"

constexpr int TileM = 32;
constexpr int TileN = 32;
constexpr int HalfN = 8;

namespace traits {
    using namespace cute;
    using type            = half;
    using pointer         = type*;
    using const_pointer   = const type*;

    using g2s_copy_atom   = Copy_Atom<UniversalCopy<uint128_t>, half>;
    using G2SCopy         = decltype(make_tiled_copy(
        g2s_copy_atom{},
        make_layout(Shape<Int<TileM>, Int<TileN / HalfN>>{}, Stride<Int<TileN / HalfN>, _1>{}),
        make_layout(Shape<_1, Int<HalfN>>{})
    ));

    using s2g_copy_atom   = Copy_Atom<UniversalCopy<uint128_t>, half>;
    using S2GCopy         = decltype(make_tiled_copy(
        s2g_copy_atom{},
        make_layout(Shape<Int<TileM>, Int<TileN / HalfN>>{}, Stride<Int<TileN / HalfN>, _1>{}),
        make_layout(Shape<_1, Int<HalfN>>{})
    ));
}

__global__ void copy_kernel(traits::const_pointer Sptr, traits::pointer Dptr, int m, int n) {
    using namespace traits;
    __shared__ half shm[TileM * TileN];

    Tensor S   = make_tensor(make_gmem_ptr(Sptr), make_shape(m, n), make_stride(n, Int<1>{}));
    Tensor D   = make_tensor(make_gmem_ptr(Dptr), make_shape(m, n), make_stride(n, Int<1>{}));
    Tensor Shm = make_tensor(make_smem_ptr(shm),  Shape<Int<TileM>, Int<TileN>>{}, Stride<Int<TileN>, _1>{});

    Tensor gS = local_tile(S, Shape<Int<TileM>, Int<TileN>>{}, make_coord(blockIdx.y, blockIdx.x));
    Tensor gD = local_tile(D, Shape<Int<TileM>, Int<TileN>>{}, make_coord(blockIdx.y, blockIdx.x));

    G2SCopy g2s_copy;
    auto this_g2s_copy = g2s_copy.get_slice(threadIdx.x);
    Tensor g2s_src = this_g2s_copy.partition_S(gS);
    Tensor g2s_dst = this_g2s_copy.partition_D(Shm);

    S2GCopy s2g_copy;
    auto this_s2g_copy = s2g_copy.get_slice(threadIdx.x);
    Tensor s2g_src = this_s2g_copy.partition_S(Shm);
    Tensor s2g_dst = this_s2g_copy.partition_D(gD);

    copy(g2s_copy, g2s_src, g2s_dst);
    copy(s2g_copy, s2g_src, s2g_dst);
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

    dim3 block(cute::size(G2SCopy{}));
    dim3 grid((n + TileN - 1) / TileN, (m + TileM - 1) / TileM);

    std::cout << "Launching Copy kernel with grid: (" << grid.x << ", " << grid.y
              << "), block: " << block.x << std::endl;
    std::cout << "Tile sizes: M=" << TileM << ", N=" << TileN << std::endl;

    copy_kernel<<<grid, block>>>(
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