// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <array>
#include <cstddef>

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

void FaceReconstruction(benchmark::State& state)
{
    int const nx = novapp::int_cast<int>(state.range());
    int const ny = nx;
    int const nz = nx;

    double const xmin = 0;
    double const xmax = 1;
    double const ymin = 0;
    double const ymax = 1;
    double const zmin = 0;
    double const zmax = 1;

    std::array<int, 3> const Nx_glob_ng {nx, ny, nz};
    std::array<int, 3> const mpi_dims_cart {0, 0, 0};
    int const Ng = 1;

    novapp::Grid grid(Nx_glob_ng, mpi_dims_cart, Ng);
    novapp::Regular const grid_type(std::array {xmin, ymin, zmin}, std::array {xmax, ymax, zmax});

    novapp::KDV_double_1d x_glob("x_glob", grid.Nx_glob_ng[0] + 2 * grid.Nghost[0] + 1);
    novapp::KDV_double_1d y_glob("y_glob", grid.Nx_glob_ng[1] + 2 * grid.Nghost[1] + 1);
    novapp::KDV_double_1d z_glob("z_glob", grid.Nx_glob_ng[2] + 2 * grid.Nghost[2] + 1);
    grid_type.execute(grid.Nghost, grid.Nx_glob_ng, x_glob.view_host(), y_glob.view_host(), z_glob.view_host());
    novapp::modify_host(x_glob, y_glob, z_glob);
    novapp::sync_device(x_glob, y_glob, z_glob);
    grid.set_grid(x_glob.view_device(), y_glob.view_device(), z_glob.view_device());

    novapp::KV_double_3d const rho("rho", grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2]);
    novapp::KV_double_5d const rho_rec("rho_rec", grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, novapp::ndim);

    Kokkos::deep_copy(rho, 1);
    Kokkos::deep_copy(rho_rec, -1);

    novapp::Minmod const limiter;
    novapp::LimitedLinearReconstruction const face_reconstruction(limiter);
    novapp::Range const range = grid.range.no_ghosts();
    Kokkos::fence();
    for ([[maybe_unused]] auto _ : state) {
        face_reconstruction.execute(range, grid, rho, rho_rec);
        Kokkos::fence();
    }

    std::size_t const cells = (static_cast<std::size_t>(nx) * ny) * nz;

    set_constant_cells_processed(state, cells);

    set_constant_bytes_processed(state, sizeof(double) * (1 + novapp::ndim * 2) * cells);
}

} // namespace

BENCHMARK(FaceReconstruction)->DenseRange(8, 63, 8)->DenseRange(64, 320, 32);
