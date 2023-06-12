//!
//! @file gravity.hpp
//!

#pragma once

#include "kokkos_shortcut.hpp"
#include "ndim.hpp"
#include "grid.hpp"

namespace novapp
{

class UniformGravity
{
private :
    KV_double_1d m_g;

public :
    explicit UniformGravity(
        KV_double_1d g)
        : m_g(g)
    {
    }

    KOKKOS_INLINE_FUNCTION
    double operator()(
            [[maybe_unused]] int i,
            [[maybe_unused]] int j,
            [[maybe_unused]] int k,
            int dir,
            [[maybe_unused]] Grid const& grid) const
    {
        return m_g(dir);
    }
}; 

/* class PointMassGravity
{
public :
    double operator()(
            int i,
            int j,
            int k, 
            int dir,
            Grid const& grid) const final
    {
        auto const x_d = grid.x.d_view;
        auto const y_d = grid.y.d_view;
        auto const z_d = grid.z.d_view;
        double r = Kokkos::sqrt(x_d[i] * x_d[i] + y_d[i] * y_d[i] + z_d[i] * z_d[i]);
    }
}; */

} // namespace novapp
