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
    double eint_min = 0;

    Kokkos::parallel_reduce(
        "check_internal_energy",
        cell_mdrange(range),
        KOKKOS_LAMBDA(int i, int j, int k, double& local_min)
        {
            EulerCons cons;
            cons.rho = rho(i, j, k);
            for (int idim = 0; idim < ndim; ++idim)
            {
                cons.rhou[idim] = rhou(i, j, k, idim);
            }
            cons.E = E(i, j, k);

            double e = compute_eint(cons);

            local_min = Kokkos::min(local_min, e);
        },
        Kokkos::Min<double>(eint_min));

        MPI_Allreduce(MPI_IN_PLACE, &eint_min, 1, MPI_DOUBLE, MPI_MIN, grid.comm_cart);
        return eint_min;
}

} // namespace novapp