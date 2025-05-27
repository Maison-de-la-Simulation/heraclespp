// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file kronecker.hpp
//!

#pragma once

#include <Kokkos_Core.hpp>

namespace novapp {

KOKKOS_FORCEINLINE_FUNCTION
constexpr int kron(int a, int b) noexcept
{
    if (a == b) {
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

} // namespace novapp
