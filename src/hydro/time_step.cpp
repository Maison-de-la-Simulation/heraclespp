#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <eos.hpp>
#include <geom.hpp>
#include <grid.hpp>
#include <kronecker.hpp>
#include <ndim.hpp>
#include <range.hpp>

#include "time_step.hpp"

namespace novapp
{

double time_step(
    Range const& range,
    EOS const& eos,
    Grid const& grid,
    double const cfl,
    KV_cdouble_3d const rho,
    KV_cdouble_4d const u,
    KV_cdouble_3d const P)
{
    Kokkos::Array<KV_cdouble_1d, 3> const dx {grid.dx, grid.dy, grid.dz};

    double inverse_dt = 0;

    Kokkos::parallel_reduce(
        "time_step",
        cell_mdrange(range),
        KOKKOS_LAMBDA(int i, int j, int k, double& local_inverse_dt)
        {
            Kokkos::Array<int, 3> const ijk {i,j,k};
            double const sound = eos.compute_speed_of_sound(rho(i, j, k), P(i, j, k));
            Kokkos::Array<double, 3> dx_geom {0, 0, 0};
            if (geom == Geometry::Geom_cartesian)
            {
                for(int idim = 0; idim < ndim; ++idim)
                {
                    dx_geom[idim] = dx[idim](ijk[idim]);
                }
            }
            if (geom == Geometry::Geom_spherical)
            {
                dx_geom[0] = grid.dx(i);
                if (ndim == 3)
                {
                    dx_geom[1] = grid.x_center(i) * grid.dy(j);
                    dx_geom[2] = grid.x_center(i) * Kokkos::sin(grid.y_center(j)) * grid.dz(k);
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

    return cfl / inverse_dt;
}

} // namespace novapp
