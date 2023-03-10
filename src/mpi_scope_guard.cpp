
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
