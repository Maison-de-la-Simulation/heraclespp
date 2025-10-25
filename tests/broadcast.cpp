// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <string>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <broadcast.hpp>
#include <kokkos_shortcut.hpp>
#include <ndim.hpp>
#include <range.hpp>

#include "utils_dual_view.hpp"

TEST(BroadcastScalar, Case1d)
{
    if (novapp::ndim != 1) {
        GTEST_SKIP();
    }

    int const nx = 10;
    int const ny = 11;
    int const nz = 12;
    int const ng = 1;
    double const outer_value = 0.;
    double const inner_value = 5.5;

    int const nx_wg = nx + (2 * ng);
    int const ny_wg = ny;
    int const nz_wg = nz;

    novapp::Range const rng({0, 0, 0}, {nx, ny, nz}, ng);
    novapp::KDV_double_3d array("array", nx_wg, ny_wg, nz_wg);
    Kokkos::deep_copy(array.view_device(), outer_value);
    array.modify_device();
    novapp::broadcast(rng.no_ghosts(), inner_value, array.view_device());
    array.modify_device();
    array.sync_host();
    {
        auto const array_h = novapp::view_host(array);
        for (int i = 0; i < nx_wg; ++i) {
            for (int j = 0; j < ny_wg; ++j) {
                for (int k = 0; k < nz_wg; ++k) {
                    if ((i == 0 || i == nx_wg - 1)) {
                        EXPECT_DOUBLE_EQ(array_h(i, j, k), outer_value);
                    } else {
                        EXPECT_DOUBLE_EQ(array_h(i, j, k), inner_value);
                    }
                }
            }
        }
    }
}

TEST(BroadcastScalar, Case2d)
{
    if (novapp::ndim != 2) {
        GTEST_SKIP();
    }

    int const nx = 10;
    int const ny = 11;
    int const nz = 12;
    int const ng = 1;
    double const outer_value = 0.;
    double const inner_value = 5.5;

    int const nx_wg = nx + (2 * ng);
    int const ny_wg = ny + (2 * ng);
    int const nz_wg = nz;

    novapp::Range const rng({0, 0, 0}, {nx, ny, nz}, ng);
    novapp::KDV_double_3d array("array", nx_wg, ny_wg, nz_wg);
    Kokkos::deep_copy(array.view_device(), outer_value);
    array.modify_device();
    novapp::broadcast(rng.no_ghosts(), inner_value, array.view_device());
    array.modify_device();
    array.sync_host();
    {
        auto const array_h = novapp::view_host(array);
        for (int i = 0; i < nx_wg; ++i) {
            for (int j = 0; j < ny_wg; ++j) {
                for (int k = 0; k < nz_wg; ++k) {
                    if ((i == 0 || i == nx_wg - 1) || (j == 0 || j == ny_wg - 1)) {
                        EXPECT_DOUBLE_EQ(array_h(i, j, k), outer_value);
                    } else {
                        EXPECT_DOUBLE_EQ(array_h(i, j, k), inner_value);
                    }
                }
            }
        }
    }
}

TEST(BroadcastScalar, Case3d)
{
    if (novapp::ndim != 3) {
        GTEST_SKIP();
    }

    int const nx = 10;
    int const ny = 11;
    int const nz = 12;
    int const ng = 1;
    double const outer_value = 0.;
    double const inner_value = 5.5;

    int const nx_wg = nx + (2 * ng);
    int const ny_wg = ny + (2 * ng);
    int const nz_wg = nz + (2 * ng);

    novapp::Range const rng({0, 0, 0}, {nx, ny, nz}, ng);
    novapp::KDV_double_3d array("array", nx_wg, ny_wg, nz_wg);
    Kokkos::deep_copy(array.view_device(), outer_value);
    array.modify_device();
    novapp::broadcast(rng.no_ghosts(), inner_value, array.view_device());
    array.modify_device();
    array.sync_host();
    {
        auto const array_h = novapp::view_host(array);
        for (int i = 0; i < nx_wg; ++i) {
            for (int j = 0; j < ny_wg; ++j) {
                for (int k = 0; k < nz_wg; ++k) {
                    if ((i == 0 || i == nx_wg - 1) || (j == 0 || j == ny_wg - 1)
                        || (k == 0 || k == nz_wg - 1)) {
                        EXPECT_DOUBLE_EQ(array_h(i, j, k), outer_value);
                    } else {
                        EXPECT_DOUBLE_EQ(array_h(i, j, k), inner_value);
                    }
                }
            }
        }
    }
}

TEST(BroadcastArray, Case1d)
{
    if (novapp::ndim != 1) {
        GTEST_SKIP();
    }

    int const nx = 10;
    int const ny = 11;
    int const nz = 12;
    int const ng = 1;
    double const outer_value = 0.;
    double const inner_value = 5.5;

    int const nx_wg = nx + (2 * ng);
    int const ny_wg = ny;
    int const nz_wg = nz;

    novapp::Range const rng({0, 0, 0}, {nx, ny, nz}, ng);
    novapp::KDV_double_3d array("array", nx_wg, ny_wg, nz_wg);
    Kokkos::deep_copy(array.view_device(), outer_value);
    array.modify_device();
    novapp::KV_double_1d const arr("arr", nx);
    Kokkos::deep_copy(arr, inner_value);
    novapp::broadcast(rng.no_ghosts(), arr, array.view_device());
    array.modify_device();
    array.sync_host();
    {
        auto const array_h = novapp::view_host(array);
        for (int i = 0; i < nx_wg; ++i) {
            for (int j = 0; j < ny_wg; ++j) {
                for (int k = 0; k < nz_wg; ++k) {
                    if ((i == 0 || i == nx_wg - 1)) {
                        EXPECT_DOUBLE_EQ(array_h(i, j, k), outer_value);
                    } else {
                        EXPECT_DOUBLE_EQ(array_h(i, j, k), inner_value);
                    }
                }
            }
        }
    }
}

TEST(BroadcastArray, Case2d)
{
    if (novapp::ndim != 2) {
        GTEST_SKIP();
    }

    int const nx = 10;
    int const ny = 11;
    int const nz = 12;
    int const ng = 1;
    double const outer_value = 0.;
    double const inner_value = 5.5;

    int const nx_wg = nx + (2 * ng);
    int const ny_wg = ny + (2 * ng);
    int const nz_wg = nz;

    novapp::Range const rng({0, 0, 0}, {nx, ny, nz}, ng);
    novapp::KDV_double_3d array("array", nx_wg, ny_wg, nz_wg);
    Kokkos::deep_copy(array.view_device(), outer_value);
    array.modify_device();
    novapp::KV_double_1d const arr("arr", nx);
    Kokkos::deep_copy(arr, inner_value);
    novapp::broadcast(rng.no_ghosts(), arr, array.view_device());
    array.modify_device();
    array.sync_host();
    {
        auto const array_h = novapp::view_host(array);
        for (int i = 0; i < nx_wg; ++i) {
            for (int j = 0; j < ny_wg; ++j) {
                for (int k = 0; k < nz_wg; ++k) {
                    if ((i == 0 || i == nx_wg - 1) || (j == 0 || j == ny_wg - 1)) {
                        EXPECT_DOUBLE_EQ(array_h(i, j, k), outer_value);
                    } else {
                        EXPECT_DOUBLE_EQ(array_h(i, j, k), inner_value);
                    }
                }
            }
        }
    }
}

TEST(BroadcastArray, Case3d)
{
    if (novapp::ndim != 3) {
        GTEST_SKIP();
    }

    int const nx = 10;
    int const ny = 11;
    int const nz = 12;
    int const ng = 1;
    double const outer_value = 0.;
    double const inner_value = 5.5;

    int const nx_wg = nx + (2 * ng);
    int const ny_wg = ny + (2 * ng);
    int const nz_wg = nz + (2 * ng);

    novapp::Range const rng({0, 0, 0}, {nx, ny, nz}, ng);
    novapp::KDV_double_3d array("array", nx_wg, ny_wg, nz_wg);
    Kokkos::deep_copy(array.view_device(), outer_value);
    array.modify_device();
    novapp::KV_double_1d const arr("arr", 10);
    Kokkos::deep_copy(arr, inner_value);
    novapp::broadcast(rng.no_ghosts(), arr, array.view_device());
    array.modify_device();
    array.sync_host();
    {
        auto const array_h = novapp::view_host(array);
        for (int i = 0; i < nx_wg; ++i) {
            for (int j = 0; j < ny_wg; ++j) {
                for (int k = 0; k < nz_wg; ++k) {
                    if ((i == 0 || i == nx_wg - 1) || (j == 0 || j == ny_wg - 1)
                        || (k == 0 || k == nz_wg - 1)) {
                        EXPECT_DOUBLE_EQ(array_h(i, j, k), outer_value);
                    } else {
                        EXPECT_DOUBLE_EQ(array_h(i, j, k), inner_value);
                    }
                }
            }
        }
    }
}
