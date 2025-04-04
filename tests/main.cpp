// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <mpi_scope_guard.hpp>

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    Kokkos::ScopeGuard const kokkos_scope(argc, argv);
    novapp::MpiScopeGuard const mpi_guard(argc, argv);

    return RUN_ALL_TESTS();
}
