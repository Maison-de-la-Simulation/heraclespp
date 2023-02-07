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
    for (int k=0; k<rho.extent_int(2); k++)
    {
        for (int j=0; j<rho.extent_int(1); j++)
        {
            for (int i=0; i<rho.extent_int(0); i++)
            {
                double const sound = eos.compute_speed_of_sound(rho(i,j,k),P(i,j,k));
                double const difference = std::fabs(u(i,j,k)) + sound;
                ax = std::fmax(difference, ax);
            }
        }
    }
    return cfl / (ax / dx);
}
