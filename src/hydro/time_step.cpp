#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <eos.hpp>
#include <grid.hpp>
#include <kronecker.hpp>
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

            double cell_inverse_dt = 0;
            for(int idim = 0; idim < ndim; ++idim)
            {
                cell_inverse_dt += (Kokkos::abs(u(i, j, k, idim)) + sound) / dx[idim](ijk[idim]);
            }

            local_inverse_dt = Kokkos::max(local_inverse_dt, cell_inverse_dt);
        },
        Kokkos::Max<double>(inverse_dt));

    MPI_Allreduce(MPI_IN_PLACE, &inverse_dt, 1, MPI_DOUBLE, MPI_MAX, grid.comm_cart);

    return cfl / inverse_dt;
}

} // namespace novapp
