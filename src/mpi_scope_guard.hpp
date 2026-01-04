// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file mpi_scope_guard.hpp
//! MPI initialization
//!

#pragma once

namespace hclpp {

class MpiScopeGuard
{
public:
    MpiScopeGuard() noexcept;

    MpiScopeGuard(int& argc, char**& argv) noexcept;

    MpiScopeGuard(MpiScopeGuard const& rhs) = delete;

    MpiScopeGuard(MpiScopeGuard&& rhs) noexcept = delete;

    ~MpiScopeGuard() noexcept;

    auto operator=(MpiScopeGuard const& rhs) -> MpiScopeGuard& = delete;

    auto operator=(MpiScopeGuard&& rhs) noexcept -> MpiScopeGuard& = delete;
};

} // namespace hclpp
