#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <PerfectGas.hpp>

#include "cfl_cond.hpp"
#include "euler_equations.hpp"
#include "range.hpp"

namespace novapp
{

double time_step(
    Range const& range,
    double const cfl,
    Kokkos::View<const double***> rho,
    Kokkos::View<const double****> u,
    Kokkos::View<const double***> P,
    Kokkos::View<const double*> dx, 
    thermodynamics::PerfectGas const& eos)
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
            dt_loc_inverse += (std::fabs(u(i, j, k, idim)) + sound) / dx(idim);
        }
        local_a = std::fmax(dt_loc_inverse, local_a);
    },
    Kokkos::Max<double>(inverse_dt));
    double result =  cfl * 1 / inverse_dt;
    MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    return result;
}

} // namespace novapp
