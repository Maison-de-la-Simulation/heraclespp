//!
//! @file gravity.hpp
//!

#pragma once

#include <units.hpp>

#include "../mesh/grid.hpp"
#include "kokkos_shortcut.hpp"
#include "ndim.hpp"
#include "nova_params.hpp"

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

    KOKKOS_FORCEINLINE_FUNCTION
    double operator()(
            [[maybe_unused]] int i,
            [[maybe_unused]] int j,
            [[maybe_unused]] int k,
            int dir) const
    {
        return m_g(dir);
    }
};

inline UniformGravity make_uniform_gravity(
        Param const& param)
{
    KDV_double_1d g_array_dv("g_array", 3);
    g_array_dv.h_view(0) = param.gx;
    g_array_dv.h_view(1) = param.gy;
    g_array_dv.h_view(2) = param.gz;
    g_array_dv.modify_host();
    g_array_dv.sync_device();
    return UniformGravity(g_array_dv.d_view);
};

class PointMassGravity
{
private :
    KV_double_1d m_g;

public :
    explicit PointMassGravity(
        KV_double_1d g)
        : m_g(g)
    {
    }

    KOKKOS_INLINE_FUNCTION
    double operator()(
            int i,
            [[maybe_unused]] int j,
            [[maybe_unused]] int k, 
            [[maybe_unused]] int dir) const
    {
        return m_g(i);
    }
};

inline PointMassGravity make_point_mass_gravity(
        Param const& param,
        Grid const& grid)
{
    KDV_double_1d g_array_dv("g_array", grid.Nx_local_wg[0]);
    auto xc = grid.x_center;
    double M = param.M;
    for (int i = 0; i < grid.Nx_local_wg[0]; ++i)
    {
        g_array_dv.h_view(i) = - units::G * M / (xc(i) * xc(i));
    }
    g_array_dv.modify_host();
    g_array_dv.sync_device();
    return PointMassGravity(g_array_dv.d_view);
};

} // namespace novapp
