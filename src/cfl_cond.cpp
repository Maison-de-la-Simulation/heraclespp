#include <Kokkos_Core.hpp>
#include <PerfectGas.hpp>

#include "cfl_cond.hpp"

double time_step(
    double const cfl,
    Kokkos::View<double***> rho,
    Kokkos::View<double***> u,
    Kokkos::View<double***> P,
    double const dx,
    [[maybe_unused]] double const dy,
    [[maybe_unused]] double const dz, 
    thermodynamics::PerfectGas const& eos)
{
    double sound;
    double difference;
    double ax = 0;

    for (int i=0; i<rho.extent_int(0); i++)
    {
        sound = eos.compute_speed_of_sound(rho(i,0,0),P(i,0,0));
        difference = std::abs(u(i,0,0)) + sound;
        ax = std::max(difference, ax);
    }
    return cfl / (ax / dx);
}
