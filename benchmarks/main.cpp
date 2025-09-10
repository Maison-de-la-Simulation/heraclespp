// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <mpi.h>

#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>

int main(int argc, char** argv)
{
    ::Kokkos::ScopeGuard const scope(argc, argv);
    MPI_Init(&argc, &argv);
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    MPI_Finalize();
    return 0;
}
