/**
 * @file main.cpp
 * Code d'hydrodynamique radiative en devenir...
 */

#include <iostream>

#include <inih/INIReader.hpp>

#include <Kokkos_Core.hpp>

#include "global_var.hpp"
#include "conv.hpp"
#include "init.hpp"
#include "fonc.hpp"
#include "slope.hpp"
#include "flux.hpp"
#include "solver.hpp"
#include "io.hpp"

#include <pdi.h>

int main(int argc, char** argv)
{
    PC_tree_t conf = PC_parse_path(argv[1]);
    PDI_init(PC_get(conf, ".pdi"));

    Kokkos::ScopeGuard guard;

    if (argc < 2)
    {
        std::cout << "usage: " << argv[0] << " <path to the ini file>\n";
        return EXIT_FAILURE;
    }

    INIReader reader(argv[1]);

    int const nx = reader.GetInteger("Grid", "nx", 10); // Cell number
    double const dx = 1. / nx;
    int inter = nx / 2; // Interface position
    int nt = 10000;
    double dt = 0.001;
    double timeout = 0.2;

    Kokkos::View<double*> x("x", nx+2); // Position

    Kokkos::parallel_for(
            "Initialisation_x",
            Kokkos::RangePolicy<>(1, nx+1),
            KOKKOS_LAMBDA(int i)
    {
        x(i) = (i - 1) * dx + dx / 2;
    });
    

/*
    for (int i = 0; i < nx+1; ++i)
    {
        std::printf("%f \n", x(i));
    }
*/
    //------------------------------------------------------------------------//

    Kokkos::View<double*> rho("rho", nx+4); // Density
    Kokkos::View<double*> rhou("rhou", nx+4); // Momentum
    Kokkos::View<double*> E("E", nx+4); // Energy
    Kokkos::View<double*> u("u", nx+4); // Speed
    Kokkos::View<double*> P("P", nx+4); // Pressure

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

    Kokkos::View<double*> FinterRho("FinterRho", nx+2); // Flux interface for density
    Kokkos::View<double*> FinterRhou("FinterRhou", nx+2); // Flux interface for momentum
    Kokkos::View<double*> FinterE("FinterE", nx+2); // Flux interface for energy

    //------------------------------------------------------------------------//

    ShockTubeInit(rho, u, P, nx+4, inter); // Initialisation (rho, u, P)

    ConvPrimCons(rho, rhou, E, u, P, nx+4, GV::gamma); // Initialisation (rho, rhou, E)

    Kokkos::deep_copy(rho_host, rho);
    Kokkos::deep_copy(u_host, u);
    Kokkos::deep_copy(P_host, P);
    Kokkos::deep_copy(rhou_host, rhou);
    Kokkos::deep_copy(E_host, E);
/*
    for (int i = 0; i < nx; ++i)
    {
        ConvPtoC convPtoC(rho_host(i), u_host(i), P_host(i));
        ConvCtoP convCtoP(rho_host(i), rhou_host(i), E_host(i));
        std::printf("%f %f %f %f %f %f %f %f %f\n", rho_host(i), u_host(i), P_host(i), rhou_host(i), E_host(i), convPtoC.ConvRhou(), convPtoC.ConvE(), convCtoP.ConvU() , convCtoP.ConvP());
        Flux flux(rho_host(i), u_host(i), P_host(i));
        std::printf("%f %f %f\n", flux.FluxRho(), flux.FluxRhou(), flux.FluxE());
    }

    for (int i = 1; i < nx-1; ++i)
    {
        Slope slope(rho_host(i-1), rho_host(i), rho_host(i+1));
        std::printf("%f %f %f\n", slope.VanLeer(), slope.Minmod(), slope.VanAlbada());
    }

    WaveSpeed WS(rho_host(0), u_host(0), P_host(0), rho_host(1), u_host(1), P_host(1));
    std::printf("%f %f\n", WS.SL(), WS.SR());

    SolverHLL solverHLL(rho_host(0), rhou_host(0), E_host(0), rho_host(1), rhou_host(1), E_host(1));
    std::printf("%f %f %f\n", solverHLL.FinterRho(), solverHLL.FinterRhou(), solverHLL.FinterE());
*/

    //------------------------------------------------------------------------//
    Kokkos::View<double*> rhoL("rhoL", nx+4);
    Kokkos::View<double*> uL("uL", nx+4);
    Kokkos::View<double*> PL("PL", nx+4);
    Kokkos::View<double*> rhoR("rhoR", nx+4);
    Kokkos::View<double*> uR("uR", nx+4);
    Kokkos::View<double*> PR("PR", nx+4);
    Kokkos::View<double*> rhouL("rhouL", nx+4);
    Kokkos::View<double*> EL("EL", nx+4);
    Kokkos::View<double*> rhouR("rhouR", nx+4);
    Kokkos::View<double*> ER("ER", nx+4);

    Kokkos::View<double*> rho_moyL("rhomoyL", nx+4);
    Kokkos::View<double*> rhou_moyL("rhoumoyL", nx+4);
    Kokkos::View<double*> E_moyL("EmoyL", nx+4);
    Kokkos::View<double*> rho_moyR("rhomoyR", nx+4);
    Kokkos::View<double*> rhou_moyR("rhoumoyR", nx+4);
    Kokkos::View<double*> E_moyR("EmoyR", nx+4);

    Kokkos::View<double*> rho_new("rhonew", nx+4);
    Kokkos::View<double*> rhou_new("rhounew", nx+4);
    Kokkos::View<double*> E_new("Enew", nx+4);

    double t = 0;
    while (t < timeout)
    {
        for (int i = 1; i < nx+3; ++i)
        {
            Slope slopeRho(rho_host(i-1), rho_host(i), rho_host(i+1));
            rhoL(i) = rho_host(i) - (dx / 2) * slopeRho.VanLeer();
            rhoR(i) = rho_host(i) + (dx / 2) * slopeRho.VanLeer();
            Slope slopeU(u_host(i-1), u_host(i), u_host(i+1));
            uL(i) = u_host(i) - (dx / 2) * slopeU.VanLeer();
            uR(i) = u_host(i) + (dx / 2) * slopeU.VanLeer();
            Slope slopeP(P_host(i-1), P_host(i), P_host(i+1));
            PL(i) = P_host(i) - (dx / 2) * slopeP.VanLeer();
            PR(i) = P_host(i) + (dx / 2) * slopeP.VanLeer();
        }
        t = t + dt;
        //std::printf("%f \n", t);

        ConvPrimCons(rhoL, rhouL, EL, uL, PL, nx+4, GV::gamma);
        ConvPrimCons(rhoR, rhouR, ER, uR, PR, nx+4, GV::gamma);

        for (int i = 1; i < nx+3; ++i)
        {
            Flux fluxL(rhoL(i), uL(i), PL(i));
            Flux fluxR(rhoR(i), uR(i), PR(i));
            rho_moyL(i) = rhoL(i) + dt / (2 * dx) * (fluxL.FluxRho() - fluxR.FluxRho());
            rhou_moyL(i) = rhouL(i) + dt / (2 * dx) * (fluxL.FluxRhou() - fluxR.FluxRhou());
            E_moyL(i) = EL(i) + dt / (2 * dx) * (fluxL.FluxE() - fluxR.FluxE());
            rho_moyR(i) = rhoR(i) + dt / (2 * dx) * (fluxL.FluxRho() - fluxR.FluxRho());
            rhou_moyR(i) = rhouR(i) + dt / (2 * dx) * (fluxL.FluxRhou() - fluxR.FluxRhou());
            E_moyR(i) = ER(i) + dt / (2 * dx) * (fluxL.FluxE() - fluxR.FluxE());
        }

        for (int i = 2; i < nx+2; i++)
        {
            SolverHLL FluxM1(rho_host(i-1), rhou_host(i-1), E_host(i-1), rho_host(i), rhou_host(i), E_host(i));
            SolverHLL FluxP1(rho_host(i), rhou_host(i), E_host(i), rho_host(i+1), rhou_host(i+1), E_host(i+1));

            rho_new(i) = rho(i) + (dt / dx) * (FluxM1.FinterRho() - FluxP1.FinterRho());
            rhou_new(i) = rhou(i) + (dt / dx) * (FluxM1.FinterRhou() -  FluxP1.FinterRhou());
            E_new(i) = E(i) + (dt / dx) * (FluxM1.FinterE() -  FluxP1.FinterE());
          /* code */
        }

        //Boundaries conditions


//  std::printf("%f %f %f\n", solverHLL.FinterRho(), solverHLL.FinterRhou(), solverHLL.FinterE());

  }

    for (int i = 0; i < nx+4; ++i)
    {
        //std::printf("%f %f %f %f\n", rhoL(i), rhoR(i), rhouL(i), EL(i));
        std::printf("%f %f\n", rho(i), rho_new(i));
    }

    write(nx, rho_new.data());

    PDI_finalize();
    PC_tree_destroy(&conf);
    std::printf("%s\n", "---Fin du programme---");
    return 0;
}
