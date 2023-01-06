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

//extern double global_a;

void ShockTubeInit(
        Kokkos::View<double*> const rho,
        Kokkos::View<double*> const u,
        Kokkos::View<double*> const P,
        int const nx,
        int const inter)
{
    Kokkos::parallel_for(
            "ShockTubeInit",
            nx,
            KOKKOS_LAMBDA(int i) {
              if(i < inter)
              {
                rho(i) = rhoL;
                u(i) = uL;
                P(i) = PL;
              }
              else
              {
                rho(i) = rhoR;
                u(i) = uR;
                P(i) =  PR;
              }
            });
}
