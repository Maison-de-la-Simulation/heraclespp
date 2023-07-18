//!
//! @file kronecker.hpp
//!

#pragma once

#include <Kokkos_Core.hpp>

namespace novapp
{

static constexpr int nfx = 1;

KOKKOS_FORCEINLINE_FUNCTION
constexpr int kron(int a, int b) noexcept
{
    if (a == b)
    {
        return 1;
    }

    return 0;
}

KOKKOS_FORCEINLINE_FUNCTION
constexpr Kokkos::Array<int, 3> lindex(int a, int i0, int i1, int i2) noexcept
{
    return Kokkos::Array<int, 3> {i0 - kron(a, 0), i1 - kron(a, 1), i2 - kron(a, 2)};
}

KOKKOS_FORCEINLINE_FUNCTION
constexpr Kokkos::Array<int, 3> rindex(int a, int i0, int i1, int i2) noexcept
{
    return Kokkos::Array<int, 3> {i0 + kron(a, 0), i1 + kron(a, 1), i2 + kron(a, 2)};
}

#define NOVA_FORCEUNROLL _Pragma("unroll")
// #define NOVA_FORCEUNROLL _Pragma("omp unroll")
// #define NOVA_FORCEUNROLL _Pragma("omp unroll full")

template <class F>
void my_parallel_for(Kokkos::Array<int, 3> const begin,
                     Kokkos::Array<int, 3> const end, F &&f) {
    // using policy = Kokkos::MDRangePolicy<
    //     Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>;
    // Kokkos::parallel_for(policy(begin, end), f);
    int const begin_x = begin[0];
    int const begin_y = begin[1];
    int const begin_z = begin[2];
    int const end_x = end[0];
    int const end_y = end[1];
    int const end_z = end[2];
// #pragma ivdep
// #pragma GCC ivdep
// #pragma clang loop vectorize(assume_safety)
#pragma omp parallel for collapse(2)
    for (int k = begin_z; k < end_z; ++k) {
        for (int j = begin_y; j < end_y; ++j) {
#pragma omp simd
          for (int i = begin_x; i < end_x; ++i) {
            f(i, j, k);
          }
        }
    }
}

} // namespace novapp
