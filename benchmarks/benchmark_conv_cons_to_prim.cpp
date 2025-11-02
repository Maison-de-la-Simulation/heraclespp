// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cstddef>

#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>
#include <array_conversion.hpp>
#include <int_cast.hpp>
#include <kokkos_shortcut.hpp>
#include <ndim.hpp>
#include <perfect_gas.hpp>
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

void cons_to_prim(benchmark::State& state)
{
    int const nx0 = novapp::int_cast<int>(state.range());
    int const nx1 = nx0;
    int const nx2 = nx0;

    novapp::Range const range({0, 0, 0}, {nx0, nx1, nx2}, 0);
    novapp::thermodynamics::PerfectGas const eos(2, 1);
    novapp::KV_double_3d const rho("rho", nx0, nx1, nx2);
    novapp::KV_double_4d const rhou("rhou", nx0, nx1, nx2, novapp::ndim);
    novapp::KV_double_3d const E("E", nx0, nx1, nx2);
    novapp::KV_double_4d const u("u", nx0, nx1, nx2, novapp::ndim);
    novapp::KV_double_3d const P("P", nx0, nx1, nx2);

    Kokkos::deep_copy(rho, 1);
    Kokkos::deep_copy(rhou, 0);
    Kokkos::deep_copy(E, 1);

    Kokkos::fence();
    for ([[maybe_unused]] auto _ : state) {
        novapp::conv_cons_to_prim(range, eos, rho, rhou, E, u, P);
        Kokkos::fence();
    }

    std::size_t const cells = (static_cast<std::size_t>(nx0) * nx1) * nx2;

    set_constant_cells_processed(state, cells);
    set_constant_bytes_processed(state, sizeof(double) * ((2 + novapp::ndim) + (1 + novapp::ndim)) * cells);
}

} // namespace

BENCHMARK(cons_to_prim)->DenseRange(8, 63, 8)->DenseRange(64, 320, 32);
