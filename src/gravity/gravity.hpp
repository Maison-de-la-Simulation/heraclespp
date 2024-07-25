//!
//! @file gravity.hpp
//!

#pragma once

#include <algorithm>
#include <stdexcept>
#include <grid.hpp>
#include <kokkos_shortcut.hpp>
#include <ndim.hpp>
#include <nova_params.hpp>
#include <units.hpp>

namespace novapp
{

class UniformGravity
{
private :
    KV_cdouble_1d m_g;

public :
    explicit UniformGravity(KV_cdouble_1d g)
        : m_g(std::move(g))
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
}

class PointMassGravity
{
private :
    KV_cdouble_1d m_g;

public :
    explicit PointMassGravity(KV_cdouble_1d g)
        : m_g(std::move(g))
    {
    }

    KOKKOS_FORCEINLINE_FUNCTION
    double operator()(
            int i,
            [[maybe_unused]] int j,
            [[maybe_unused]] int k,
            int dir) const
    {
        if (dir == 0)
        {
            return m_g(i);
        }
        return 0;
    }
};

inline PointMassGravity make_point_mass_gravity(
    Param const& param,
    Grid const& grid)
{
    KV_double_1d const g_array("g_array", grid.Nx_local_wg[0]);
    double const M = param.M;

    KV_cdouble_1d const xc = grid.x_center;

    Kokkos::parallel_for(
        "point_mass_gravity",
        Kokkos::RangePolicy<int>(0, grid.Nx_local_wg[0]),
        KOKKOS_LAMBDA(int i)
        {
            g_array(i) = - units::G * M / (xc(i) * xc(i));
        });
    return PointMassGravity(g_array);
}

class InternalMassGravity
{
private :
    KV_cdouble_1d m_g;

public :
    explicit InternalMassGravity(KV_cdouble_1d g)
        : m_g(std::move(g))
    {
    }

    KOKKOS_FORCEINLINE_FUNCTION
    double operator()(
            int i,
            [[maybe_unused]] int j,
            [[maybe_unused]] int k,
            int dir) const
    {
        if (dir == 0)
        {
            return m_g(i);
        }
        return 0;
    }
};

inline InternalMassGravity make_internal_mass_gravity(
    Param const& param,
    Grid const& grid,
    KV_cdouble_3d const& rho)
{
    if (grid.Ncpu != 1)
    {
        throw std::runtime_error("The function make_internal_mass_gravity does not handle more than 1 MPI process");
    }

    KV_double_1d g_array_dv("g_array", grid.Nx_local_wg[0]);
    KV_double_1d dv_total("dv_mean_array", grid.Nx_local_ng[0] + grid.Nghost[0]); // sum dv
    KV_double_1d rho_mean("rho_mean_array", grid.Nx_local_ng[0] + grid.Nghost[0]); // sum (rho * dv) / (sum dv)

    KV_double_1d M_r("M_r_array", grid.Nx_local_wg[0]); // total mass at r
    double const M_star = param.M;
    auto const x = grid.x;
    auto const xc = grid.x_center;
    auto const dv = grid.dv;

    Kokkos::Array<int, 3> nghost;
    std::copy(grid.Nghost.begin(), grid.Nghost.end(), nghost.data());

    for (int i = 0; i < grid.Nx_local_ng[0] + grid.Nghost[0]; ++i)
    {
        Kokkos::parallel_reduce(
            "integration_shell",
            Kokkos::MDRangePolicy<int, Kokkos::Rank<2>>
            ({0, 0},
            {grid.Nx_local_ng[1], grid.Nx_local_ng[2]}),
            KOKKOS_LAMBDA(int j, int k, double& local_sum_mass, double& local_sum_dv)
            {
                int offset_i = i + nghost[0];
                int offset_j = j + nghost[1];
                int offset_k = k + nghost[2];
                local_sum_mass += rho(offset_i, offset_j, offset_k) * dv(offset_i, offset_j, offset_k);
                local_sum_dv += dv(offset_i, offset_j, offset_k);
            },
            Kokkos::Sum(Kokkos::subview(rho_mean, i)), Kokkos::Sum(Kokkos::subview(dv_total, i)));
    }

    Kokkos::parallel_for(
        "rho_mean_o_dv_tot",
        Kokkos::RangePolicy<int>(0, grid.Nx_local_ng[0] + grid.Nghost[0]),
        KOKKOS_LAMBDA(int i)
        {
            rho_mean(i) /= dv_total(i);
        });

    Kokkos::parallel_for(
        "ghost_cells_Mr",
        Kokkos::RangePolicy<int>(0, grid.Nghost[0]),
        KOKKOS_LAMBDA(int i)
        {
            M_r(i) = M_star;
        });

    Kokkos::parallel_scan(
        "mass_mid_cell",
        Kokkos::RangePolicy<int>(grid.Nghost[0], grid.Nx_local_ng[0] + 2 * grid.Nghost[0]),
        KOKKOS_LAMBDA(int i, double& partial_sum, bool is_final)
        {
            int offset  = i - nghost[0];
            double x3 = x(i) * x(i) * x(i);
            if (i == nghost[0])
            {
                partial_sum += 4. / 3 * units::pi * (xc(i) * xc(i) * xc(i) - x3) * rho_mean(offset);
            }
            else
            {
                partial_sum +=  4. / 3 * units::pi * (x3 - xc(i-1) * xc(i-1) * xc(i-1)) * rho_mean(offset-1)
                    + 4. / 3 * units::pi * (xc(i) * xc(i) * xc(i) - x3) * rho_mean(offset);
            }
            if (is_final)
            {
                M_r(i) = partial_sum;
            }
        });

    Kokkos::parallel_for(
        "internal_mass_gravity",
        Kokkos::RangePolicy<int>(0, grid.Nx_local_wg[0]),
        KOKKOS_LAMBDA(int i)
        {
            g_array_dv(i) = - units::G * M_r(i) / (xc(i) * xc(i));
        });

    return InternalMassGravity(g_array_dv);
}

} // namespace novapp
