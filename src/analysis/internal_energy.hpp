//!
//! @file internal_energy.hpp
//!

#pragma once

#include <grid.hpp>
#include <kokkos_shortcut.hpp>
#include <ndim.hpp>
#include <range.hpp>

#include "euler_equations.hpp"

namespace novapp
{

inline double internal_energy(
    Range const& range,
    Grid const& grid,
    KV_cdouble_3d const rho,
    KV_cdouble_4d const rhou,
    KV_cdouble_3d const E)
{
    double evol = 0;

    Kokkos::parallel_reduce(
        "min_internal_energy",
        cell_mdrange(range),
        KOKKOS_LAMBDA(int i, int j, int k, double& local_evol)
        {
            EulerCons cons;
            cons.rho = rho(i, j, k);
            for (int idim = 0; idim < ndim; ++idim)
            {
                cons.rhou[idim] = rhou(i, j, k, idim);
            }
            cons.E = E(i, j, k);

            double const cell_evol = compute_evol(cons);

            local_evol = Kokkos::min(local_evol, cell_evol);
        },
        Kokkos::Min<double>(evol));

    MPI_Allreduce(MPI_IN_PLACE, &evol, 1, MPI_DOUBLE, MPI_MIN, grid.comm_cart);

    return evol;
}

} // namespace novapp