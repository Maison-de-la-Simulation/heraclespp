//!
//! @file kronecker.hpp
//!
#pragma once

constexpr int kron(
    int a, 
    int b) noexcept
{
    if (a == b)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}