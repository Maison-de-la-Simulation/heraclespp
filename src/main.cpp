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
#include "flux.hpp"
#include "solver.hpp"
#include "io.hpp"
#include "face_reconstruction.hpp"
// #include "coordinate_system.hpp"
#include "boundary.hpp"

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

    double const dx = 1. / (grid.Nx_glob[0]+2*grid.Nghost);
    int inter = grid.Nx_glob[0] / 2; // Interface position
    double cfl = 0.4;

    init_write(max_iter, output_frequency, grid.Nghost);

    std::string const reconstruction_type = reader.Get("hydro", "reconstruction", "Minmod");
    std::unique_ptr<IFaceReconstruction> face_reconstruction
            = factory_face_reconstruction(reconstruction_type, dx);

    int alpha;
    std::string const system_choice = reader.Get("Grid", "system", "Cartesian");
    std::printf("%s\n", system_choice.c_str());
    if (system_choice == "Cartesian")
    {
        alpha = 0;
    }
    else if (system_choice == "Cylindrical")
    {
        alpha = 1;
    }
     else if (system_choice == "Spherical")
    {
        alpha = 2;
    }
    std::printf("%d\n", alpha);

    Kokkos::View<double*> r("r", grid.Nx_glob[0]+2*grid.Nghost); // Position

    Kokkos::parallel_for(
            "Initialisation_x",
            Kokkos::RangePolicy<>(0, grid.Nx_glob[0]+2*grid.Nghost),
            KOKKOS_LAMBDA(int i)
    {
        r(i) = i * dx + dx / 2;
    });
/*
    for (int i = 0; i < grid.Nx_glob[0]+2*grid.Nghost; ++i)
    {
        std::printf("*d %f \n", i, r(i));
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
    /*
    for (int i = 0; i < grid.Nx_glob[0]+2*grid.Nghost; ++i)
    {
        std::printf("%d %f %f %f %f %f\n",i , rho(i), u(i), P(i), rhou(i), E(i));
    }
    */
    double t = 0;
    int iter = 0;
    bool make_output = false;
    
    while (t <= timeout & iter<=max_iter)
    {
        face_reconstruction->execute(rho, rhoL, rhoR); // Calcul des pentes
        face_reconstruction->execute(u, uL, uR);
        face_reconstruction->execute(P, PL, PR);
       
        double dt = Dt(rhoL, uL, PL, rhoR, uR, PR, dx, cfl);
        //std::printf("dt=%f\n", dt);

        ConvPrimCons(rhoL, rhouL, EL, uL, PL, GV::gamma); // Conversion en variables conservatives
        ConvPrimCons(rhoR, rhouR, ER, uR, PR, GV::gamma);

        double dto2dx = dt / (2 * dx);

        Kokkos::parallel_for(
            "Extrapolation",
            Kokkos::RangePolicy<>(1, grid.Nx_glob[0]+3),
            KOKKOS_LAMBDA(int i)
        {
            Flux fluxL(rhoL(i), uL(i), PL(i));
            Flux fluxR(rhoR(i), uR(i), PR(i));

            rho_moyL(i) = rhoL(i) + dto2dx * (fluxL.FluxRho() - fluxR.FluxRho());
            rhou_moyL(i) = rhouL(i) + dto2dx * (fluxL.FluxRhou() - fluxR.FluxRhou());
            E_moyL(i) = EL(i) + dto2dx * (fluxL.FluxE() - fluxR.FluxE());
            rho_moyR(i) = rhoR(i) + dto2dx * (fluxL.FluxRho() - fluxR.FluxRho());
            rhou_moyR(i) = rhouR(i) + dto2dx * (fluxL.FluxRhou() - fluxR.FluxRhou());
            E_moyR(i) = ER(i) + dto2dx * (fluxL.FluxE() - fluxR.FluxE());
        });

         double dtodx = dt / dx;

         Kokkos::parallel_for(
            "New value",
            Kokkos::RangePolicy<>(2, grid.Nx_glob[0]+2),
            KOKKOS_LAMBDA(int i)
        {
            double dv = (1. / (1 + alpha)) * (std::pow(r(i), alpha + 1) - std::pow(r(i-1), alpha + 1));
            double rm1 = std::pow(r(i-1), alpha);
            double rp1 = std::pow(r(i), alpha);
            double dtodv = dt / dv;

            SolverHLL FluxM1(rho_moyR(i-1), rhou_moyR(i-1), E_moyR(i-1), rho_moyL(i), rhou_moyL(i), E_moyL(i));
            SolverHLL FluxP1(rho_moyR(i), rhou_moyR(i), E_moyR(i),rho_moyL(i+1), rhou_moyL(i+1), E_moyL(i+1));

            rho_new(i) = rho(i) + dtodv * (rm1 * FluxM1.FinterRho() - rp1 * FluxP1.FinterRho());
            rhou_new(i) = rhou(i) + dtodv * (rm1 * FluxM1.FinterRhou() - rp1 * FluxP1.FinterRhou()) + dtodv * (rp1 *  PR(i) - rm1 * PL(i)) - dtodx *(PR(i) - PL(i));
            E_new(i) = E(i) + dtodv * (rm1 * FluxM1.FinterE() - rp1 * FluxP1.FinterE());
            /*
            rho_new(i) = rho(i) + dtodx * (FluxM1.FinterRho() - FluxP1.FinterRho());
            rhou_new(i) = rhou(i) + dtodx * (FluxM1.FinterRhou() -  FluxP1.FinterRhou());
            E_new(i) = E(i) + dtodx * (FluxM1.FinterE() -  FluxP1.FinterE());
            */
        });
        
        //Boundary condition
        rho_new(0) = rho_new(1)=  rho_new(2);
        rhou_new(0) = rhou_new(1) = rhou_new(2);
        E_new(0) = E_new(1) = E_new(2);

        rho_new(grid.Nx_glob[0]+grid.Nghost) = rho_new(grid.Nx_glob[0]+grid.Nghost+1) = rho_new(grid.Nx_glob[0]+1); 
        rhou_new(grid.Nx_glob[0]+grid.Nghost) = rhou_new(grid.Nx_glob[0]+grid.Nghost+1) = rhou_new(grid.Nx_glob[0]+1); 
        E_new(grid.Nx_glob[0]+grid.Nghost) = E_new(grid.Nx_glob[0]+grid.Nghost+1) = E_new(grid.Nx_glob[0]+1); 
        
        //GradientNull(rho_new, rhou_new, E_new, grid.Nx_glob[0]);
       
        ConvConsPrim(rho_new, rhou_new, E_new, u, P, GV::gamma); //Conversion des variables conservatives en primitives
        Kokkos::deep_copy(rho, rho_new);
        Kokkos::deep_copy(rhou, rhou_new);
        Kokkos::deep_copy(E, E_new);

        if (t + dt > timeout)
        {
            dt = timeout - t + 0.00001 ;
        }
        //std::printf("dt = %f\n", dt);
        for (int i = 0; i < grid.Nx_glob[0]+2*grid.Nghost; ++i)
        {
        //std::printf("fin boucle %d %f %f %f %f %f\n", i, rho(i), u(i), P(i), rhou(i), E(i));
        }
        
        make_output = should_output(iter, output_frequency, max_iter, t, dt, timeout);
        if(make_output)
        {
            write(iter, grid.Nx_glob[0], t, rho.data(), u.data());
        }
        

        
        //write(iter, grid.Nx_glob[0], rho.data());
        std::printf("Time = %f et iteration = %d  \n", t, iter);
        t = t + dt;
        iter++;
    }
    std::printf("Time = %f et iteration = %d  \n", t, iter);
        
    

    PDI_finalize();
    PC_tree_destroy(&conf);

    std::printf("%s\n", "---Fin du programme---");
    return 0;
}
