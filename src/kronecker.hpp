//!
//! @file kronecker.hpp
//!
#pragma once

#include <Kokkos_Core.hpp>

namespace novapp
{

KOKKOS_INLINE_FUNCTION
constexpr int kron(int a, int b) noexcept
{
    if (a == b)
    {
        return 1;
    }

    return 0;
}

} // namespace novapp
