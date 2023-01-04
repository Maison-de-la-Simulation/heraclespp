#include <Kokkos_Core.hpp>

#include "init.hpp"

void ShockTubeInit(
        Kokkos::View<double*> const rho,
        Kokkos::View<double*> const rhou,
        Kokkos::View<double*> const E,
        int const inter,
        int const nx)
{
    // Left side
    double const rhoL = 1; // Density
    double const uL = 0; // Speed
    double const PL = 1; // Pressure
    // Right side
    double const rhoR = 0.125;
    double const uR = 0;
    double const PR = 0.1;

    Kokkos::parallel_for(
            "ShockTubeInit",
            nx,
            KOKKOS_LAMBDA(int i) { rho(i) = rhoL; });
}
