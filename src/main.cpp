/**
 * @file main.cpp
 * Code d'hydrodynamique radiative en devenir...
 */

#include <iostream>
#include <memory>

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
#include "face_reconstruction.hpp"

#include <pdi.h>

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cout << "usage: " << argv[0] << " <path to the ini file> <path to the yaml file>\n";
        return EXIT_FAILURE;
    }

    Kokkos::ScopeGuard guard;

    INIReader reader(argv[1]);

    PC_tree_t conf = PC_parse_path(argv[2]);
    PDI_init(PC_get(conf, ".pdi"));

    Grid grid(1, 0);

    grid.Nx_glob[0] = reader.GetInteger("Grid", "Nx_glob", 10); // Cell number
    grid.Nx_glob[1] = reader.GetInteger("Grid", "Ny_glob", 10); // Cell number
    grid.Nx_glob[2] = reader.GetInteger("Grid", "Nz_glob", 10); // Cell number
    double const timeout = reader.GetReal("Run", "timeout", 0.2);
    int const max_iter = reader.GetInteger("Output", "max_iter", 10000);
    int const output_frequency = reader.GetInteger("Output", "frequency", 10);
    double const dx = 1. / grid.Nx_glob[0];
    int inter = grid.Nx_glob[0] / 2; // Interface position
    int nt = 10000;
    double dt = 0.001;

    init_write(max_iter, output_frequency);

    std::string const reconstruction_type = reader.Get("hydro", "reconstruction", "Minmod");
    std::unique_ptr<IFaceReconstruction> face_reconstruction
            = factory_face_reconstruction(reconstruction_type, dx);

    Kokkos::View<double*> x("x", grid.Nx_glob[0]+2); // Position

    Kokkos::parallel_for(
            "Initialisation_x",
            Kokkos::RangePolicy<>(1, grid.Nx_glob[0]+1),
            KOKKOS_LAMBDA(int i)
    {
        x(i) = (i - 1) * dx + dx / 2;
    });
    

/*
    for (int i = 0; i < Nx_glob+1; ++i)
    {
        std::printf("%f \n", x(i));
    }
*/
    //------------------------------------------------------------------------//
    

    Kokkos::View<double*> rho("rho", grid.Nx_glob[0]+2*grid.Nghost); // Density
    Kokkos::View<double*> rhou("rhou", grid.Nx_glob[0]+2*grid.Nghost); // Momentum
    Kokkos::View<double*> E("E", grid.Nx_glob[0]+2*grid.Nghost); // Energy
    Kokkos::View<double*> u("u", grid.Nx_glob[0]+2*grid.Nghost); // Speed
    Kokkos::View<double*> P("P", grid.Nx_glob[0]+2*grid.Nghost); // Pressure

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

    Kokkos::View<double*> FinterRho("FinterRho", grid.Nx_glob[0]+2); // Flux interface for density
    Kokkos::View<double*> FinterRhou("FinterRhou", grid.Nx_glob[0]+2); // Flux interface for momentum
    Kokkos::View<double*> FinterE("FinterE", grid.Nx_glob[0]+2); // Flux interface for energy

    //------------------------------------------------------------------------//

    ShockTubeInit(rho, u, P, grid.Nx_glob[0]+2*grid.Nghost, inter); // Initialisation (rho, u, P)

    ConvPrimCons(rho, rhou, E, u, P, GV::gamma); // Initialisation (rho, rhou, E)

    Kokkos::deep_copy(rho_host, rho);
    Kokkos::deep_copy(u_host, u);
    Kokkos::deep_copy(P_host, P);
    Kokkos::deep_copy(rhou_host, rhou);
    Kokkos::deep_copy(E_host, E);
/*
    for (int i = 0; i < Nx_glob; ++i)
    {
        ConvPtoC convPtoC(rho_host(i), u_host(i), P_host(i));
        ConvCtoP convCtoP(rho_host(i), rhou_host(i), E_host(i));
        std::printf("%f %f %f %f %f %f %f %f %f\n", rho_host(i), u_host(i), P_host(i), rhou_host(i), E_host(i), convPtoC.ConvRhou(), convPtoC.ConvE(), convCtoP.ConvU() , convCtoP.ConvP());
        Flux flux(rho_host(i), u_host(i), P_host(i));
        std::printf("%f %f %f\n", flux.FluxRho(), flux.FluxRhou(), flux.FluxE());
    }

    for (int i = 1; i < Nx_glob-1; ++i)
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
    
    Kokkos::View<double*> rhoL("rhoL", grid.Nx_glob[0]+2*grid.Nghost);
    Kokkos::View<double*> uL("uL", grid.Nx_glob[0]+2*grid.Nghost);
    Kokkos::View<double*> PL("PL", grid.Nx_glob[0]+2*grid.Nghost);
    Kokkos::View<double*> rhoR("rhoR", grid.Nx_glob[0]+2*grid.Nghost);
    Kokkos::View<double*> uR("uR", grid.Nx_glob[0]+2*grid.Nghost);
    Kokkos::View<double*> PR("PR", grid.Nx_glob[0]+2*grid.Nghost);
    Kokkos::View<double*> rhouL("rhouL", grid.Nx_glob[0]+2*grid.Nghost);
    Kokkos::View<double*> EL("EL", grid.Nx_glob[0]+2*grid.Nghost);
    Kokkos::View<double*> rhouR("rhouR", grid.Nx_glob[0]+2*grid.Nghost);
    Kokkos::View<double*> ER("ER", grid.Nx_glob[0]+2*grid.Nghost);

    Kokkos::View<double*> rho_moyL("rhomoyL", grid.Nx_glob[0]+2*grid.Nghost);
    Kokkos::View<double*> rhou_moyL("rhoumoyL", grid.Nx_glob[0]+2*grid.Nghost);
    Kokkos::View<double*> E_moyL("EmoyL", grid.Nx_glob[0]+2*grid.Nghost);
    Kokkos::View<double*> rho_moyR("rhomoyR", grid.Nx_glob[0]+2*grid.Nghost);
    Kokkos::View<double*> rhou_moyR("rhoumoyR", grid.Nx_glob[0]+2*grid.Nghost);
    Kokkos::View<double*> E_moyR("EmoyR", grid.Nx_glob[0]+2*grid.Nghost);

    Kokkos::View<double*> rho_new("rhonew", grid.Nx_glob[0]+2*grid.Nghost);
    Kokkos::View<double*> rhou_new("rhounew", grid.Nx_glob[0]+2*grid.Nghost);
    Kokkos::View<double*> E_new("Enew", grid.Nx_glob[0]+2*grid.Nghost);

    double t = 0;
    int iter = 0;
    

    while (t < timeout || iter<=max_iter)
    {
        face_reconstruction->execute(rho, rhoL, rhoR);
        face_reconstruction->execute(u, uL, uR);
        face_reconstruction->execute(P, PL, PR);

        t = t + dt;
        //std::printf("%f \n", t);

        ConvPrimCons(rhoL, rhouL, EL, uL, PL, GV::gamma);
        ConvPrimCons(rhoR, rhouR, ER, uR, PR, GV::gamma);

        Kokkos::parallel_for(
            "Extrapolation",
            Kokkos::RangePolicy<>(1, grid.Nx_glob[0]+3),
            KOKKOS_LAMBDA(int i)
        {
            Flux fluxL(rhoL(i), uL(i), PL(i));
            Flux fluxR(rhoR(i), uR(i), PR(i));
            rho_moyL(i) = rhoL(i) + dt / (2 * dx) * (fluxL.FluxRho() - fluxR.FluxRho());
            rhou_moyL(i) = rhouL(i) + dt / (2 * dx) * (fluxL.FluxRhou() - fluxR.FluxRhou());
            E_moyL(i) = EL(i) + dt / (2 * dx) * (fluxL.FluxE() - fluxR.FluxE());
            rho_moyR(i) = rhoR(i) + dt / (2 * dx) * (fluxL.FluxRho() - fluxR.FluxRho());
            rhou_moyR(i) = rhouR(i) + dt / (2 * dx) * (fluxL.FluxRhou() - fluxR.FluxRhou());
            E_moyR(i) = ER(i) + dt / (2 * dx) * (fluxL.FluxE() - fluxR.FluxE());
        });

        double dtdx = dt / dx;

        Kokkos::parallel_for(
            "New value",
            Kokkos::RangePolicy<>(grid.Nghost, grid.Nx_glob[0]+grid.Nghost),
            KOKKOS_LAMBDA(int i)
        {
            SolverHLL FluxM1(rho_host(i-1), rhou_host(i-1), E_host(i-1), rho_host(i), rhou_host(i), E_host(i));
            SolverHLL FluxP1(rho_host(i), rhou_host(i), E_host(i), rho_host(i+1), rhou_host(i+1), E_host(i+1));

            rho_new(i) = rho(i) + dtdx * (FluxM1.FinterRho() - FluxP1.FinterRho());
            rhou_new(i) = rhou(i) + dtdx * (FluxM1.FinterRhou() -  FluxP1.FinterRhou());
            E_new(i) = E(i) + dtdx * (FluxM1.FinterE() -  FluxP1.FinterE());
          /* code */
        });

        //Boundaries conditions
    write(iter, grid.Nx_glob[0], rho_new.data());
    iter++;
//  std::printf("%f %f %f\n", solverHLL.FinterRho(), solverHLL.FinterRhou(), solverHLL.FinterE());

  }

    for (int i = 0; i < grid.Nx_glob[0]+2*grid.Nghost; ++i)
    {
        //std::printf("%f %f %f %f\n", rhoL(i), rhoR(i), rhouL(i), EL(i));
        // std::printf("%f %f\n", rho(i), rho_new(i));
    }

    

    PDI_finalize();
    PC_tree_destroy(&conf);
    std::printf("%s\n", "---Fin du programme---");
    return 0;
}
