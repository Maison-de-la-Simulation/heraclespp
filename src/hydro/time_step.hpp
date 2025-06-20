// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file time_step.hpp
//! CFL condition
//!

#pragma once

#include "concepts.hpp"
#include <mpi.h>

#include <cassert>

#include <Kokkos_Core.hpp>
#include <geom.hpp>
#include <grid.hpp>
#include <kokkos_shortcut.hpp>
#include <ndim.hpp>
#include <range.hpp>

namespace novapp
{

//! Time step with the cfl condition
//! @param[in] range output iteration range
//! @param[in] eos equation of state
//! @param[in] grid mesh metadata
//! @param[in] rho density array 3D
//! @param[in] u velocity array 3D
//! @param[in] P pressure array 3D
//! @return time step
template <concepts::EulerEoS EoS>
[[nodiscard]] double time_step(
    Range const& range,
    EoS const& eos,
    Grid const& grid,
    KV_cdouble_3d const& rho,
    KV_cdouble_4d const& u,
    KV_cdouble_3d const& P)
{
    assert(equal_extents({0, 1, 2}, rho, u, P));
    assert(u.extent(3) == ndim);

    auto const& dx = grid.dx;
    auto const& dy = grid.dy;
    auto const& dz = grid.dz;
    auto const& x_center = grid.x_center;
    auto const& y_center = grid.y_center;

    double inverse_dt = 0;

    Kokkos::parallel_reduce(
        "time_step",
        cell_mdrange(range),
        KOKKOS_LAMBDA(int i, int j, int k, double& local_inverse_dt)
        {
            double const sound = eos.compute_speed_of_sound(rho(i, j, k), P(i, j, k));
            Kokkos::Array<double, 3> dx_geom {dx(i), dy(j), dz(k)};
            if (geom == Geometry::Geom_spherical)
            {
                if (ndim == 2)
                {
                    dx_geom[1] *= x_center(i);
                }
                if (ndim == 3)
                {
                    dx_geom[1] *= x_center(i);
                    dx_geom[2] *= x_center(i) * Kokkos::sin(y_center(j));
                }
            }

            double cell_inverse_dt = 0;
            for(int idim = 0; idim < ndim; ++idim)
            {
                cell_inverse_dt += (Kokkos::abs(u(i, j, k, idim)) + sound) / dx_geom[idim];
            }
            local_inverse_dt = Kokkos::max(local_inverse_dt, cell_inverse_dt);
        },
        Kokkos::Max<double>(inverse_dt));

    MPI_Allreduce(MPI_IN_PLACE, &inverse_dt, 1, MPI_DOUBLE, MPI_MAX, grid.comm_cart);

    return 1 / inverse_dt;
}

} // namespace novapp
