// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <mpi_scope_guard.hpp>

auto main(int argc, char** argv) -> int
{
    ::testing::InitGoogleTest(&argc, argv);

    Kokkos::ScopeGuard const kokkos_scope(argc, argv);
    hclpp::MpiScopeGuard const mpi_guard(argc, argv);

    return RUN_ALL_TESTS();
}
