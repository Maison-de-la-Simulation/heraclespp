// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <array>
#include <cstddef>
#include <string>

#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>
#include <grid.hpp>
#include <grid_type.hpp>
#include <int_cast.hpp>
#include <kokkos_shortcut.hpp>
#include <limited_linear_reconstruction.hpp>
#include <ndim.hpp>
#include <range.hpp>
#include <slope_limiters.hpp>

namespace {

void set_constant_bytes_processed(benchmark::State& state, std::size_t const bytes)
{
    state.counters["bytes_per_second"] = benchmark::Counter(static_cast<double>(bytes), benchmark::Counter::kIsIterationInvariantRate);
}

void set_constant_cells_processed(benchmark::State& state, std::size_t const cells)
{
    state.counters["cells_per_second"] = benchmark::Counter(static_cast<double>(cells), benchmark::Counter::kIsIterationInvariantRate);
}

void face_reconstruction(benchmark::State& state)
{
    int const nx0 = hclpp::int_cast<int>(state.range());
    int const nx1 = nx0;
    int const nx2 = nx0;

    double const x0min = 0;
    double const x0max = 1;
    double const x1min = 0;
    double const x1max = 1;
    double const x2min = 0;
    double const x2max = 1;

    std::array<int, 3> const Nx_glob_ng {nx0, nx1, nx2};
    std::array<int, 3> const mpi_dims_cart {0, 0, 0};
    int const Ng = 1;

    hclpp::Grid grid(Nx_glob_ng, mpi_dims_cart, Ng);
    hclpp::Regular const regular_grid(std::array {x0min, x1min, x2min}, std::array {x0max, x1max, x2max});

    hclpp::KDV_double_1d x0_glob("x0_glob", grid.Nx_glob_ng[0] + (2 * grid.Nghost[0]) + 1);
    hclpp::KDV_double_1d x1_glob("x1_glob", grid.Nx_glob_ng[1] + (2 * grid.Nghost[1]) + 1);
    hclpp::KDV_double_1d x2_glob("x2_glob", grid.Nx_glob_ng[2] + (2 * grid.Nghost[2]) + 1);
    regular_grid.execute(grid.Nghost, grid.Nx_glob_ng, x0_glob.view_host(), x1_glob.view_host(), x2_glob.view_host());
    hclpp::modify_host(x0_glob, x1_glob, x2_glob);
    hclpp::sync_device(x0_glob, x1_glob, x2_glob);
    grid.set_grid(x0_glob.view_device(), x1_glob.view_device(), x2_glob.view_device());

    hclpp::KV_double_3d const rho("rho", grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2]);
    hclpp::KV_double_5d const rho_rec("rho_rec", grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, hclpp::ndim);

    Kokkos::deep_copy(rho, 1);
    Kokkos::deep_copy(rho_rec, -1);

    hclpp::Minmod const limiter;
    hclpp::LimitedLinearReconstruction const face_reconstruction(limiter);
    hclpp::Range const range = grid.range.no_ghosts();
    Kokkos::fence();
    for ([[maybe_unused]] auto _ : state) {
        face_reconstruction.execute(range, grid, rho, rho_rec);
        Kokkos::fence();
    }

    std::size_t const cells = (static_cast<std::size_t>(nx0) * nx1) * nx2;

    set_constant_cells_processed(state, cells);

    set_constant_bytes_processed(state, sizeof(double) * (1 + hclpp::ndim * 2) * cells);
}

} // namespace

BENCHMARK(face_reconstruction)->DenseRange(8, 63, 8)->DenseRange(64, 320, 32);
