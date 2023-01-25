#include <Kokkos_Core.hpp>

#include "cfl_cond.hpp"
#include "speed_sound.hpp"

double time_step(
    double const cfl,
    Kokkos::View<double***> rho,
    Kokkos::View<double***> u,
    Kokkos::View<double***> P,
    double const dx,
    double const dy,
    double const dz, 
    double const gamma)
{
    double sound;
    double difference;
    double ax = 0;

    for (int i=0; i<rho.extent(0); i++)
    {
        sound = speed_sound2(rho(i,0,0),P(i,0,0), gamma);
        difference = std::abs(u(i,0,0)) + sound;
        ax = std::max(difference, ax);
    }
    return cfl / (ax / dx);
}