#include <Kokkos_Core.hpp>

#include "init.hpp"

// Global variables
// Left side
double const rhoL = 1; // Density
double const uL = 0; // Speed
double const PL = 1; // Pressure
// Right side
double const rhoR = 0.125;
double const uR = 0;
double const PR = 0.1;

void ShockTubeInit(
    Kokkos::View<double***> const rho,
    Kokkos::View<double***> const u,
    Kokkos::View<double***> const P,
    int const inter)
{
    Kokkos::parallel_for(
       "ShockTubeInit",
       Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
       {0, 0, 0},
       {rho.extent(0), rho.extent(1), rho.extent(2)}),
       KOKKOS_LAMBDA(int i, int j, int k)
    {
        if (i <= inter)
        {
            rho(i, j, k) = rhoL;
            u(i, j, k) = uL;
            P(i, j, k) = PL; 
        }
        else
        {
            rho(i, j, k) = rhoR;
            u(i, j, k) = uR;
            P(i, j, k) =  PR;
        }
    });
}
