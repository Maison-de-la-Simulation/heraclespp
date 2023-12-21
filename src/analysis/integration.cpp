//!
//! @file integration.hpp
//!

#include <grid.hpp>
#include <kokkos_shortcut.hpp>
#include <range.hpp>

#include "integration.hpp"

namespace novapp
{

double integrate(
    Range const& range,
    Grid const& grid,
    KV_cdouble_3d const var)
{
    double total_mass = 0;

    Kokkos::parallel_reduce(
        "integration",
        cell_mdrange(range),
        KOKKOS_LAMBDA(int i, int j, int k, double& local_sum)
        {
            local_sum += var(i, j, k) * grid.dv(i, j, k);
        },
        Kokkos::Sum<double>(total_mass));

    MPI_Allreduce(MPI_IN_PLACE, &total_mass, 1, MPI_DOUBLE, MPI_SUM, grid.comm_cart);
    return total_mass;
}

} // namespace novapp