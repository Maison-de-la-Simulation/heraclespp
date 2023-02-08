#include <Kokkos_Core.hpp>
#include <PerfectGas.hpp>

#include "cfl_cond.hpp"

double time_step(
    double const cfl,
    Kokkos::View<const double***> rho,
    Kokkos::View<const double***> u,
    Kokkos::View<const double***> P,
    double const dx,
    [[maybe_unused]] double const dy,
    [[maybe_unused]] double const dz, 
    thermodynamics::PerfectGas const& eos)
{
    double ax = 0;
    Kokkos::parallel_reduce(
            "time_step",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {0, 0, 0},
            {rho.extent_int(0), rho.extent_int(1), rho.extent_int(2)}),
            KOKKOS_LAMBDA(int i, int j, int k, double& local_ax) {
                double const sound = eos.compute_speed_of_sound(rho(i, j, k), P(i, j, k));
                double const difference = std::fabs(u(i, j, k)) + sound;
                local_ax = std::fmax(difference, local_ax);
            },
            Kokkos::Max<double>(ax));
    return cfl / (ax / dx);
}
