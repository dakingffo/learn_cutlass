#include "header.hpp"

void composition() {
    using namespace cute;
    auto layout_A = make_layout(Shape<_8, _8>{}, Stride<_8, _1>{});
    auto layout_B = make_layout(Shape<_4, _8>{}, Stride<_1, _4>{});
    auto layout_C = composition(layout_A, layout_B);

    std::cout << "Layout A (Base): "; print(layout_A); std::cout << std::endl;
    print_layout(layout_A);
    std::cout << "Layout B (Filter): "; print(layout_B); std::cout << std::endl;
    print_layout(layout_B);
    std::cout << "Layout C (Composed): "; print(layout_C); std::cout << std::endl;
    print_layout(layout_C);


    auto layoutShared0 = make_layout(Shape<_8, _8>{}, Stride<_8, _1>{});
    auto swizzle0 = composition(Swizzle<3, 0, 3>{}, layoutShared0);
    print_layout(swizzle0);

    auto layoutShared1 = make_layout(Shape<_8, _16>{}, Stride<_8, _1>{}); // if element is half, 8 bank(4bytes)
    auto swizzle1 = composition(Swizzle<3, 1, 3>{}, layoutShared1);
    print_layout(swizzle1);
}