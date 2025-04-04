// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <mpi.h>

#include "mpi_scope_guard.hpp"

namespace novapp
{

MpiScopeGuard::MpiScopeGuard() noexcept
{
    MPI_Init(nullptr, nullptr);
}

MpiScopeGuard::MpiScopeGuard(int& argc, char**& argv) noexcept
{
    MPI_Init(&argc, &argv);
}

MpiScopeGuard::~MpiScopeGuard() noexcept
{
    MPI_Finalize();
}

} // namespace novapp
