//!
//! @file conservation.hpp
//!

#pragma once

#include <grid.hpp>
#include <kokkos_shortcut.hpp>
#include <range.hpp>

namespace novapp
{

inline double conservation(
    Range const& range,
    Grid const& grid,
    KV_cdouble_3d const rho)
{
    double sum = 0;

    Kokkos::parallel_reduce(
        "check_conservation",
        cell_mdrange(range),
        KOKKOS_LAMBDA(int i, int j, int k, double& local_sum)
        {
            local_sum += rho(i, j, k) * grid.dv(i, j, k);
        },
        Kokkos::Sum<double>(sum));

    double total_mass = sum;
    MPI_Allreduce(MPI_IN_PLACE, &total_mass, 1, MPI_DOUBLE, MPI_SUM, grid.comm_cart);
    return total_mass;
}
} // namespace novapp