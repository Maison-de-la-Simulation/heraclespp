/**
 * @file mpi_scope_guard.hpp
 * MPI initialization
 */
#pragma once

class MpiScopeGuard
{
public:
    MpiScopeGuard() noexcept;

    MpiScopeGuard(int& argc, char**& argv) noexcept;

    MpiScopeGuard(MpiScopeGuard const& rhs) = delete;

    MpiScopeGuard(MpiScopeGuard&& rhs) noexcept = delete;

    ~MpiScopeGuard() noexcept;

    MpiScopeGuard& operator=(MpiScopeGuard const& rhs) = delete;

    MpiScopeGuard& operator=(MpiScopeGuard&& rhs) noexcept = delete;
};
