/**
 * @file main.cpp
 * Code d'hydrodynamique radiative en devenir...
 */

#include <iostream>

#include <Kokkos_Core.hpp>

#include "init.hpp"

int main(int argc, char** argv)
{
    Kokkos::ScopeGuard guard;

    int nx = 100; // Size
    int inter = nx / 2; // Interface

    Kokkos::View<double*> rho("rho", nx); // Density
    Kokkos::View<double*> rhou("rhou", nx); // Momentum
    Kokkos::View<double*> E("E", nx); // Energy

    Kokkos::View<double*, Kokkos::HostSpace> rho_host
            = Kokkos::create_mirror_view(rho); // Density always on host
    Kokkos::View<double*, Kokkos::HostSpace> rhou_host
            = Kokkos::create_mirror_view(rhou); // Momentum always on host
    Kokkos::View<double*, Kokkos::HostSpace> E_host
            = Kokkos::create_mirror_view(E); // Energy always on host

    ShockTubeInit(rho, rhou, E, inter, nx);

    Kokkos::deep_copy(rho_host, rho);
    Kokkos::deep_copy(rhou_host, rhou);
    Kokkos::deep_copy(E_host, E);
    for (int i = 0; i < nx; ++i)
    {
        std::printf("%f\n", rho_host(i));
    }

    return 0;
}
