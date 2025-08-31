// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file gravity.cpp
//!

#include <mpi.h>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <utility>

#include <Kokkos_Core.hpp>
#include <grid.hpp>
#include <kokkos_shortcut.hpp>
#include <units.hpp>

#include "gravity.hpp"

namespace novapp
{

UniformGravity::UniformGravity(KV_cdouble_1d g) : m_g(std::move(g)) {}

UniformGravity make_uniform_gravity(double const gx, double const gy, double const gz)
{
    KDV_double_1d g_array_dv("g_array", 3);
    {
        // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
        auto const g_array_h = g_array_dv.view_host();
        g_array_h(0) = gx;
        g_array_h(1) = gy;
        g_array_h(2) = gz;
    }
    g_array_dv.modify_host();
    g_array_dv.sync_device();
    return UniformGravity(g_array_dv.view_device());
}

PointMassGravity::PointMassGravity(KV_cdouble_1d g) : m_g(std::move(g)) {}

PointMassGravity make_point_mass_gravity(
    double const central_mass,
    Grid const& grid)
{
    KV_double_1d const g_array("g_array", grid.Nx_local_wg[0]);

    KV_cdouble_1d const xc = grid.x_center;

    Kokkos::parallel_for(
        "point_mass_gravity",
        Kokkos::RangePolicy<int>(0, grid.Nx_local_wg[0]),
        KOKKOS_LAMBDA(int i)
        {
            g_array(i) = - units::G * central_mass / (xc(i) * xc(i));
        });
    return PointMassGravity(g_array);
}

InternalMassGravity::InternalMassGravity(KV_cdouble_1d g) : m_g(std::move(g)) {}

InternalMassGravity make_internal_mass_gravity(
    double const central_mass,
    Grid const& grid,
    KV_cdouble_3d const& rho)
{
    if (grid.mpi_dims_cart[0] != 1)
    {
        throw std::runtime_error("The function make_internal_mass_gravity does not handle more than 1 MPI process in the first dimension");
    }

    KV_double_1d const g_array_dv("g_array", grid.Nx_local_wg[0]);
    KDV_double_1d dv_total_dv("dv_mean_array", grid.Nx_local_ng[0] + grid.Nghost[0]); // sum dv
    KDV_double_1d rho_mean_dv("rho_mean_array", grid.Nx_local_ng[0] + grid.Nghost[0]); // sum (rho * dv) / (sum dv)

    KV_double_1d const M_r("M_r_array", grid.Nx_local_wg[0]); // total mass at r
    auto const x = grid.x;
    auto const xc = grid.x_center;
    auto const dv = grid.dv;

    Kokkos::Array<int, 3> nghost;
    std::ranges::copy(grid.Nghost, nghost.data());

    rho_mean_dv.modify_device();
    dv_total_dv.modify_device();
    for (int i = 0; i < grid.Nx_local_ng[0] + grid.Nghost[0]; ++i)
    {
        Kokkos::parallel_reduce(
            "integration_shell",
            Kokkos::MDRangePolicy<int, Kokkos::Rank<2>>
            ({0, 0},
            {grid.Nx_local_ng[1], grid.Nx_local_ng[2]}),
            KOKKOS_LAMBDA(int j, int k, double& local_sum_mass, double& local_sum_dv)
            {
                int const offset_i = i + nghost[0];
                int const offset_j = j + nghost[1];
                int const offset_k = k + nghost[2];
                local_sum_mass += rho(offset_i, offset_j, offset_k) * dv(offset_i, offset_j, offset_k);
                local_sum_dv += dv(offset_i, offset_j, offset_k);
            },
            Kokkos::Sum(Kokkos::subview(rho_mean_dv.view_device(), i)),
            Kokkos::Sum(Kokkos::subview(dv_total_dv.view_device(), i)));
    }
    rho_mean_dv.sync_host();
    dv_total_dv.sync_host();

    rho_mean_dv.modify_host();
    dv_total_dv.modify_host();
    MPI_Allreduce(MPI_IN_PLACE, dv_total_dv.view_host().data(), dv_total_dv.extent_int(0), MPI_DOUBLE, MPI_SUM, grid.comm_cart_horizontal);
    MPI_Allreduce(MPI_IN_PLACE, rho_mean_dv.view_host().data(), rho_mean_dv.extent_int(0), MPI_DOUBLE, MPI_SUM, grid.comm_cart_horizontal);
    rho_mean_dv.sync_device();
    dv_total_dv.sync_device();

    KV_double_1d const rho_mean = rho_mean_dv.view_device();
    KV_cdouble_1d const dv_total = dv_total_dv.view_device();
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
            M_r(i) = central_mass;
        });

    Kokkos::parallel_scan(
        "mass_mid_cell",
        Kokkos::RangePolicy<int>(grid.Nghost[0], grid.Nx_local_ng[0] + 2 * grid.Nghost[0]),
        KOKKOS_LAMBDA(int i, double& partial_sum, bool is_final)
        {
            int const offset  = i - nghost[0];
            double const x3 = x(i) * x(i) * x(i);
            if (i == nghost[0])
            {
                partial_sum += central_mass + 4. / 3 * units::pi * (xc(i) * xc(i) * xc(i) - x3) * rho_mean(offset);
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
