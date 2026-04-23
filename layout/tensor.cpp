#include "header.hpp"
void tensor() {
    using namespace cute;
    constexpr int size = 8;
    int nums[8 * 8]{};
    for (int i = 0; i < size; i++) {
        std::fill(&nums[i * size], &nums[i * size] + size, i);
    }

    Tensor t0 = make_tensor((int*)nums, Shape<_8, _8>{}, Stride<_8, _1>{});
    print_tensor(t0);
    print_layout(t0.layout());

    Tensor t1 = make_tensor((int*)nums, Shape<Shape<_2, _4>, Shape<_2, _4>>{}, Stride<Stride<_2, _4>, Stride<_1, _16>>{});
    print_tensor(t1);
    print_layout(t1.layout());

    Tensor t2 = make_tensor((int*)nums, Shape<Shape<_2, _4>, Shape<_2, _4>>{}, Stride<Stride<_1, _16>, Stride<_2, _4>>{});
    print_tensor(t2);
    print_layout(t2.layout());

    Tensor t3 = make_tensor((int*)nums, Shape<Shape<_4, _2>, Shape<_4, _2>>{}, Stride<Stride<_4, _16>, Stride<_1, _32>>{});
    print_tensor(t3);
    print_layout(t3.layout());

    Tensor t4 = make_tensor((int*)nums, Shape<Shape<_4, _2>, Shape<_4, _2>>{}, Stride<Stride<_4, _16>, Stride<_4, _1>>{});
    print_tensor(t4);
    print_layout(t4.layout());

    Tensor t5 = logical_divide(t0, Shape<_2, _2>{});
    print_tensor(t5(make_coord(_, 1), make_coord(_, 1)));
    print_tensor(local_tile(t0, Shape<_2, _2>{}, make_coord(1, 1)));
    // print_layout(t5.layout());
}