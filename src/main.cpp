/**
 * @file main.cpp
 * Code d'hydrodynamique radiative en devenir...
 */

#include <iostream>

#include <Kokkos_Core.hpp>

#include "init.hpp"
#include "fonc.hpp"
#include "solver.hpp"
#include "exact.hpp"
#include "global_var.hpp"

// #include <pdi.h>

//double global_a = 10.5;

int main(int argc, char** argv)
{
    Kokkos::ScopeGuard guard;

    int nx = 100; // Size
    int inter = nx / 2; // Interface

    //double gamma = 5. / 3;

    //------------------------------------------------------------------------//

    Kokkos::View<double*> rho("rho", nx); // Density
    Kokkos::View<double*> rhou("rhou", nx); // Momentum
    Kokkos::View<double*> E("E", nx); // Energy
    Kokkos::View<double*> u("u", nx); // Speed
    Kokkos::View<double*> P("P", nx); // Pressure

    Kokkos::View<double*, Kokkos::HostSpace> rho_host
            = Kokkos::create_mirror_view(rho); // Density always on host
    Kokkos::View<double*, Kokkos::HostSpace> rhou_host
            = Kokkos::create_mirror_view(rhou); // Momentum always on host
    Kokkos::View<double*, Kokkos::HostSpace> E_host
            = Kokkos::create_mirror_view(E); // Energy always on host
    Kokkos::View<double*, Kokkos::HostSpace> u_host
            = Kokkos::create_mirror_view(rhou); // Speedalways on host
    Kokkos::View<double*, Kokkos::HostSpace> P_host
            = Kokkos::create_mirror_view(P); // Pressure always on host

    Kokkos::View<double*> FinterRho("FinterRho", nx); // Flux interface for density
    Kokkos::View<double*> FinterRhou("FinterRhou", nx); // Flux interface for momentum
    Kokkos::View<double*> FinterE("FinterE", nx); // Flux interface for energy

    //------------------------------------------------------------------------//

    ShockTubeInit(rho, u, P, nx, inter); // Initialisation of rho, u, P
    ConvPrimCons(rho, rhou, E, u, P, nx, GV::gamma); // Initialisation of rhou, E

    Kokkos::deep_copy(rho_host, rho);
    Kokkos::deep_copy(u_host, u);
    Kokkos::deep_copy(P_host, P);
    Kokkos::deep_copy(rhou_host, rhou);
    Kokkos::deep_copy(E_host, E);

    for (int i = 0; i < nx; ++i)
    {
        std::printf("%f %f %f %f %f\n", rho_host(i), u_host(i), P_host(i), rhou_host(i), E_host(i));
    }

    printf("%s\n", "---Fin du programme---");
    return 0;
}
