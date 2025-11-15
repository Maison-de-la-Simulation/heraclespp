// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <array>
#include <string>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <gravity.hpp>
#include <grid.hpp>
#include <grid_type.hpp>
#include <kokkos_shortcut.hpp>
#include <ndim.hpp>
#include <range.hpp>
#include <units.hpp>

#include "utils_dual_view.hpp"

namespace {

void test_gravity_internal_gravity()
{
    // The target function does not work in 1D and 2D
    if constexpr (novapp::ndim != 3)
    {
        GTEST_SKIP() << "Skipping test for ndim != 3";
    }

    int const nx = 15;
    int const Ng = 2;
    double const x0min = 1E3;
    double const x0max = 1E4;
    double const x1min = 0.7853981633974483;
    double const x1max = 2.356194490192345;
    double const x2min = 0.7853981633974483;
    double const x2max = 2.356194490192345;
    double const M_star = 2E30;
    double const rho_value = 1E10;

    std::array<int, 3> Nx_glob_ng {0, 0, 0};
    for (int idim = 0; idim < novapp::ndim; ++idim) {
        Nx_glob_ng[idim] = nx;
    }
    std::array<int, 3> const mpi_dims_cart {0, 0, 0};

    novapp::Grid grid(Nx_glob_ng, mpi_dims_cart, Ng);
    novapp::Regular const regular_grid(std::array {x0min, x1min, x2min}, std::array {x0max, x1max, x2max});

    novapp::KDV_double_1d x0_glob("x0_glob", grid.Nx_glob_ng[0]+(2*grid.Nghost[0])+1);
    novapp::KDV_double_1d x1_glob("x1_glob", grid.Nx_glob_ng[1]+(2*grid.Nghost[1])+1);
    novapp::KDV_double_1d x2_glob("x2_glob", grid.Nx_glob_ng[2]+(2*grid.Nghost[2])+1);
    regular_grid.execute(grid.Nghost, grid.Nx_glob_ng, x0_glob.view_host(), x1_glob.view_host(), x2_glob.view_host());
    novapp::modify_host(x0_glob, x1_glob, x2_glob);
    novapp::sync_device(x0_glob, x1_glob, x2_glob);
    grid.set_grid(x0_glob.view_device(), x1_glob.view_device(), x2_glob.view_device());

    novapp::KV_double_3d const rho("rho", grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2]);
    Kokkos::deep_copy(rho, rho_value);

    // Numerical gravitational field
    novapp::InternalMassGravity const g = novapp::make_internal_mass_gravity(M_star, grid, rho);

    // Theoretical gravitational field
    auto x0c = grid.x0_center;
    novapp::KV_double_1d const g_th("g_th", grid.Nx_local_wg[0]);
    Kokkos::deep_copy(g_th, 0.);
    Kokkos::parallel_for("", novapp::cell_mdrange(grid.range.no_ghosts()), KOKKOS_LAMBDA(int i, int, int)
    {
        double const M_i = 4. / 3 * novapp::units::pi * rho_value * (x0c(i) * x0c(i) * x0c(i) - x0min * x0min * x0min);
        g_th(i) = - novapp::units::G * (M_star + M_i) / (x0c(i) * x0c(i));
    });

    // Linf norm
    double error = 0;
    // Max abs(gth)
    double norm = 0;
    Kokkos::parallel_reduce("", novapp::cell_mdrange(grid.range.no_ghosts()), KOKKOS_LAMBDA(int i, int j, int k, double& local_error, double& local_norm)
    {
        local_error = Kokkos::max(Kokkos::abs(g_th(i) - g(i, j, k, 0)), local_error);
        local_norm = Kokkos::max(Kokkos::abs(g_th(i)), local_norm);
    }, Kokkos::Max<double>(error), Kokkos::Max<double>(norm));
    EXPECT_NEAR(error / norm, 0., 1e-14);
}

} // namespace

TEST(Gravity, InternalGravity)
{
    test_gravity_internal_gravity();
}

namespace {

void test_gravity_uniform_gravity()
{
    double const gx = 1.4;
    double const gy = 1.9;
    double const gz = 3.7;

    novapp::KDV_double_1d g("g", 3);
    {
        auto const g_h = novapp::view_host(g);
        g_h(0) = gx;
        g_h(1) = gy;
        g_h(2) = gz;
    }
    g.modify_host();
    g.sync_device();
    novapp::UniformGravity const gravity(g.view_device());
    int const nx = 5;
    int const ny = 4;
    int const nz = 3;
    novapp::KDV_double_4d result("result", nx, ny, nz, 3);
    {
        auto const result_d = novapp::view_device(result);
        Kokkos::parallel_for(
                Kokkos::MDRangePolicy<Kokkos::IndexType<int>, Kokkos::Rank<3>>({0, 0, 0}, {nx, ny, nz}),
                KOKKOS_LAMBDA(int i, int j, int k) {
                    result_d(i, j, k, 0) = gravity(i, j, k, 0);
                    result_d(i, j, k, 1) = gravity(i, j, k, 1);
                    result_d(i, j, k, 2) = gravity(i, j, k, 2);
                });
    }
    result.modify_device();
    result.sync_host();
    {
        auto const result_h = novapp::view_host(result);
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    EXPECT_EQ(result_h(i, j, k, 0), gx);
                    EXPECT_EQ(result_h(i, j, k, 1), gy);
                    EXPECT_EQ(result_h(i, j, k, 2), gz);
                }
            }
        }
    }
}

} // namespace

TEST(Gravity, UniformGravity)
{
    test_gravity_uniform_gravity();
}
