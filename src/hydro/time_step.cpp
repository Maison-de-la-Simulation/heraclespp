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
            if (geom == Geometry::Geom_cartesian)
            {
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

    return cfl / inverse_dt;
}

} // namespace novapp
