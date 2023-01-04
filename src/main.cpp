/**
 * @file main.cpp
 * Code d'hydrodynamique radiative en devenir...
 */

#include <iostream>
#include <Kokkos_Core.hpp>

#include "init.hpp"

int main(int argc, char** argv)
{
    int nx = 100; // Size
    int inter = nx / 2; // Interface

    Kokkos::ScopeGuard guard;
    Kokkos::View<double*> rho("rho", nx); /**< Density */
    Kokkos::View<double*> rhou("rhou", nx); /**< Momentum */
    Kokkos::View<double*> E("E", nx); /**< Energy */

    ShockTubeInit(rho, rhou, E, inter, nx);

    Kokkos::parallel_for("", nx, KOKKOS_LAMBDA(int i){
      printf("%f\n", rho(i));
    });

    std::cout << "Hello world\n";
    return 0;
}
