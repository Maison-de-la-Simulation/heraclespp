// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cstddef>

#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>
#include <PerfectGas.hpp>
#include <array_conversion.hpp>
#include <int_cast.hpp>
#include <kokkos_shortcut.hpp>
#include <ndim.hpp>
#include <range.hpp>

namespace {

void set_constant_bytes_processed(benchmark::State& state, std::size_t const bytes)
{
    state.counters["bytes_per_second"] = benchmark::Counter(static_cast<double>(bytes), benchmark::Counter::kIsIterationInvariantRate);
}

void set_constant_cells_processed(benchmark::State& state, std::size_t const cells)
{
    state.counters["cells_per_second"] = benchmark::Counter(static_cast<double>(cells), benchmark::Counter::kIsIterationInvariantRate);
}

void ConsToPrim(benchmark::State& state)
{
    int const nx = novapp::int_cast<int>(state.range());
    int const ny = nx;
    int const nz = nx;

    novapp::Range const range({0, 0, 0}, {nx, ny, nz}, 0);
    novapp::thermodynamics::PerfectGas const eos(2, 1);
    novapp::KV_double_3d const rho("rho", nx, ny, nz);
    novapp::KV_double_4d const rhou("rhou", nx, ny, nz, novapp::ndim);
    novapp::KV_double_3d const E("E", nx, ny, nz);
    novapp::KV_double_4d const u("u", nx, ny, nz, novapp::ndim);
    novapp::KV_double_3d const P("P", nx, ny, nz);

    Kokkos::deep_copy(rho, 1);
    Kokkos::deep_copy(rhou, 0);
    Kokkos::deep_copy(E, 1);

    Kokkos::fence();
    for ([[maybe_unused]] auto _ : state) {
        novapp::conv_cons_to_prim(range, eos, rho, rhou, E, u, P);
        Kokkos::fence();
    }

    std::size_t const cells = (static_cast<std::size_t>(nx) * ny) * nz;

    set_constant_cells_processed(state, cells);
    set_constant_bytes_processed(state, sizeof(double) * ((2 + novapp::ndim) + (1 + novapp::ndim)) * cells);
}

} // namespace

BENCHMARK(ConsToPrim)->DenseRange(8, 63, 8)->DenseRange(64, 320, 32);
