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
    KV_cdouble_3d rho,
    KV_cdouble_4d u,
    KV_cdouble_3d P)
{
    double inverse_dt = 0;

    Kokkos::parallel_reduce(
        "time_step",
        cell_mdrange(range),
        KOKKOS_LAMBDA(int i, int j, int k, double& local_inverse_dt)
        {
            double const sound = eos.compute_speed_of_sound(rho(i, j, k), P(i, j, k));

            double cell_inverse_dt = 0;
            for(int idim = 0; idim < ndim; ++idim)
            {
                double dx = kron(idim,0) * grid.dx(i) + kron(idim,1) * grid.dy(j) + kron(idim,2) * grid.dz(k);
                cell_inverse_dt += (Kokkos::fabs(u(i, j, k, idim)) + sound) / dx;
            }
            local_inverse_dt = Kokkos::fmax(cell_inverse_dt, local_inverse_dt);
        },
        Kokkos::Max<double>(inverse_dt));

    MPI_Allreduce(MPI_IN_PLACE, &inverse_dt, 1, MPI_DOUBLE, MPI_MAX, grid.comm_cart);

    return cfl / inverse_dt;
}

} // namespace novapp
