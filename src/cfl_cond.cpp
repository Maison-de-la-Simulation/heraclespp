#include <mpi.h>
#include <PerfectGas.hpp>
#include "cfl_cond.hpp"
#include "euler_equations.hpp"
#include "range.hpp"
#include "grid.hpp"

namespace novapp
{

double time_step(
    Range const& range,
    double const cfl,
    KV_cdouble_3d rho,
    KV_cdouble_4d u,
    KV_cdouble_3d P,
    thermodynamics::PerfectGas const& eos,
    Grid const& grid)
{
    double inverse_dt = 0;
    
    auto const [begin, end] = cell_range(range);
    Kokkos::parallel_reduce(
    "time_step",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
    KOKKOS_LAMBDA(int i, int j, int k, double& local_a) {
        double const sound = eos.compute_speed_of_sound(rho(i, j, k), P(i, j, k));
        double dt_loc_inverse = 0;
        for(int idim = 0; idim < ndim; idim++)
        {
            double dx = kron(idim,0) * grid.dx(i) + kron(idim,1) * grid.dy(j) + kron(idim,2) * grid.dz(k);
            dt_loc_inverse += (Kokkos::fabs(u(i, j, k, idim)) + sound) / dx;
        }
        local_a = Kokkos::fmax(dt_loc_inverse, local_a);
    },
    Kokkos::Max<double>(inverse_dt));
    double result =  cfl * 1 / inverse_dt;
    MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_DOUBLE, MPI_MIN, grid.comm_cart);
    return result;
}

} // namespace novapp
