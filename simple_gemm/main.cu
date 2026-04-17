#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

#include "../utility/timer.hpp"
#include "../utility/error_guard.hpp"

constexpr int TileM = 128;
constexpr int TileN = 128;
constexpr int TileK = 32;
    
using T          = cutlass::half_t; 
using mma_op     = cute::SM80_16x8x16_F16F16F16F16_TN;
using mma_traits = cute::MMA_Traits<mma_op>;
using mma_atom   = cute::MMA_Atom<mma_traits>;

using MMA = decltype(cute::make_tiled_mma(
    mma_atom{}, 
    cute::make_layout(cute::Shape<cute::_2, cute::_2, cute::_1>{}), 
    cute::make_layout(cute::Shape<cute::_1, cute::_2, cute::_1>{})
));

__global__ void gemm_kernel(T *Cptr, const T *Aptr, const T *Bptr, int m, int n, int k) {
    using namespace cute;
    Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

    int ix = blockIdx.x;
    int iy = blockIdx.y;

    Tensor gA = local_tile(A, make_tile(Int<TileM>{}, Int<TileK>{}), make_coord(iy, _));  // (TileM, TileK, num_tile_k)
    Tensor gB = local_tile(B, make_tile(Int<TileN>{}, Int<TileK>{}), make_coord(ix, _));  // (TileN, TileK, num_tile_k)
    Tensor gC = local_tile(C, make_tile(Int<TileM>{}, Int<TileN>{}), make_coord(iy, ix)); // (TileM, TileN)

    MMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tAgA = thr_mma.partition_A(gA); // (MMA, MMA_M, MMA_K, num_tile_k)
    auto tBgB = thr_mma.partition_B(gB); // (MMA, MMA_N, MMA_K, num_tile_k)
    auto tCgC = thr_mma.partition_C(gC); // (MMA, MMA_M, MMA_N)

    auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
    auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
    auto tCrC = thr_mma.partition_fragment_C(gC(_, _));    // (MMA, MMA_M, MMA_N)

    clear(tCrC);

    int num_tile_k = size<2>(gA);
#pragma unroll
    for (int itile = 0; itile < num_tile_k; ++itile) {
        copy(tAgA(_, _, _, itile), tArA);
        copy(tBgB(_, _, _, itile), tBrB);
        gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
    }

    copy(tCrC, tCgC);
}

int main() {
    const int m = 1024;
    const int n = 1024;
    const int k = 1024;

    thrust::host_vector<T> hA(m * k);
    thrust::host_vector<T> hB(n * k);

    thrust::default_random_engine rng(123456);
    thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);

    thrust::generate(hA.begin(), hA.end(), [&]() { return T(dist(rng)); });
    thrust::generate(hB.begin(), hB.end(), [&]() { return T(dist(rng)); });

    thrust::device_vector<T> dA = hA;
    thrust::device_vector<T> dB = hB;
    thrust::device_vector<T> dC(m * n, T(0));

    dim3 block(cute::size(MMA{}));
    dim3 grid((n + TileN - 1) / TileN, (m + TileM - 1) / TileM);

    std::cout << "Launching GEMM kernel with grid: (" << grid.x << ", " << grid.y
              << "), block: " << block.x << std::endl;
    std::cout << "Tile sizes: M=" << TileM << ", N=" << TileN << ", K=" << TileK << std::endl;

    gemm_kernel<<<grid, block>>>(
        thrust::raw_pointer_cast(dC.data()), 
        thrust::raw_pointer_cast(dA.data()), 
        thrust::raw_pointer_cast(dB.data()), 
        m, n, k
    );

    ERROR_GUARD(cudaDeviceSynchronize());

    thrust::host_vector<T> hC = dC;

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
            if (std::abs(sum - expected) > 3) {
                std::cerr << "Expect " << expected << " at [" << i << "," << j << "], but get" << sum << ".\n";
                return 0;
            }
        }
    }
    std::cout << "SUCCESS: GPU results match CPU reference!" << std::endl;

    using namespace utility;
    TIMING("SimpleGEMM", config<loop<5>>) {
        gemm_kernel<<<grid, block>>>(
            thrust::raw_pointer_cast(dC.data()), 
            thrust::raw_pointer_cast(dA.data()), 
            thrust::raw_pointer_cast(dB.data()), 
            m, n, k
        );
    }

    return 0;
}