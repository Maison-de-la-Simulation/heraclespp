/**
 * @file main.cpp
 * Code d'hydrodynamique radiative en devenir...
 */

#include <iostream>
#include <memory>

#include <inih/INIReader.hpp>

#include <Kokkos_Core.hpp>

#include "global_var.hpp"
#include "array_conversion.hpp"
#include "init.hpp"
#include "float_conversion.hpp"
#include "flux.hpp"
#include "solver.hpp"
#include "io.hpp"
#include "face_reconstruction.hpp"
#include "coordinate_system.hpp"
#include "cfl_cond.hpp"
#include "boundary.hpp"
#include "grid.hpp"
#include "PerfectGas.hpp"

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

    Grid grid;

    grid.Nx_glob[0] = reader.GetInteger("Grid", "Nx_glob", 10); // Cell number
    grid.Nx_glob[1] = reader.GetInteger("Grid", "Ny_glob", 10); // Cell number
    grid.Nx_glob[2] = reader.GetInteger("Grid", "Nz_glob", 10); // Cell number
    double const timeout = reader.GetReal("Run", "timeout", 0.2);
    int const max_iter = reader.GetInteger("Output", "max_iter", 10000);
    int const output_frequency = reader.GetInteger("Output", "frequency", 10);

    thermodynamics::PerfectGas eos(reader.GetReal("PerfectGas", "gamma", 5. / 3), 0.0);

    double const dx = 1. / (grid.Nx_glob[0]+2*grid.Nghost);
    int inter = grid.Nx_glob[0] / 2; // Interface position
    double const cfl = 0.4;

    init_write(max_iter, output_frequency, grid.Nghost);

    std::string const reconstruction_type = reader.Get("hydro", "reconstruction", "Minmod");
    std::unique_ptr<IFaceReconstruction> face_reconstruction
            = factory_face_reconstruction(reconstruction_type, dx);
    int alpha = GetenumIndex(reader.Get("Grid", "system", "Cartesian"));
    
    Kokkos::View<double***> position("position", grid.Nx_glob[0]+2*grid.Nghost, 1, 1); // Position

    Kokkos::parallel_for(
        "initialisation_r",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
        {0, 0, 0},
        {grid.Nx_glob[0]+2*grid.Nghost, 1, 1}),
        KOKKOS_LAMBDA(int i, int j, int k)
    {
        position(i, j, k) = i * dx; // Position of the left interface
    });

    Kokkos::View<double***> rho("rho", grid.Nx_glob[0]+2*grid.Nghost, 1, 1); // Density
    Kokkos::View<double***> rhou("rhou", grid.Nx_glob[0]+2*grid.Nghost, 1, 1); // Momentum
    Kokkos::View<double***> E("E", grid.Nx_glob[0]+2*grid.Nghost, 1, 1); // Energy
    Kokkos::View<double***> u("u", grid.Nx_glob[0]+2*grid.Nghost, 1, 1); // Speed
    Kokkos::View<double***> P("P", grid.Nx_glob[0]+2*grid.Nghost, 1, 1); // Pressure
    
    Kokkos::View<double***, Kokkos::HostSpace> rho_host
            = Kokkos::create_mirror_view(rho); // Density always on host
    Kokkos::View<double***, Kokkos::HostSpace> rhou_host
            = Kokkos::create_mirror_view(rhou); // Momentum always on host
    Kokkos::View<double***, Kokkos::HostSpace> E_host
            = Kokkos::create_mirror_view(E); // Energy always on host
    Kokkos::View<double***, Kokkos::HostSpace> u_host
            = Kokkos::create_mirror_view(u); // Speedalways on host
    Kokkos::View<double***, Kokkos::HostSpace> P_host
            = Kokkos::create_mirror_view(P); // Pressure always on host

    Kokkos::View<double***> rhoL("rhoL", grid.Nx_glob[0]+2*grid.Nghost, 1, 1);
    Kokkos::View<double***> uL("uL", grid.Nx_glob[0]+2*grid.Nghost, 1, 1);
    Kokkos::View<double***> PL("PL", grid.Nx_glob[0]+2*grid.Nghost, 1, 1);
    Kokkos::View<double***> rhoR("rhoR", grid.Nx_glob[0]+2*grid.Nghost, 1, 1);
    Kokkos::View<double***> uR("uR", grid.Nx_glob[0]+2*grid.Nghost, 1, 1);
    Kokkos::View<double***> PR("PR", grid.Nx_glob[0]+2*grid.Nghost, 1, 1);
    Kokkos::View<double***> rhouL("rhouL", grid.Nx_glob[0]+2*grid.Nghost, 1, 1);
    Kokkos::View<double***> EL("EL", grid.Nx_glob[0]+2*grid.Nghost, 1, 1);
    Kokkos::View<double***> rhouR("rhouR", grid.Nx_glob[0]+2*grid.Nghost, 1, 1);
    Kokkos::View<double***> ER("ER", grid.Nx_glob[0]+2*grid.Nghost, 1, 1);

    Kokkos::View<double***> rho_moyL("rhomoyL", grid.Nx_glob[0]+2*grid.Nghost, 1, 1);
    Kokkos::View<double***> rhou_moyL("rhoumoyL", grid.Nx_glob[0]+2*grid.Nghost, 1, 1);
    Kokkos::View<double***> E_moyL("EmoyL", grid.Nx_glob[0]+2*grid.Nghost, 1, 1);
    Kokkos::View<double***> rho_moyR("rhomoyR", grid.Nx_glob[0]+2*grid.Nghost, 1, 1);
    Kokkos::View<double***> rhou_moyR("rhoumoyR", grid.Nx_glob[0]+2*grid.Nghost, 1, 1);
    Kokkos::View<double***> E_moyR("EmoyR", grid.Nx_glob[0]+2*grid.Nghost, 1, 1);

    Kokkos::View<double***> rho_new("rhonew", grid.Nx_glob[0]+2*grid.Nghost, 1, 1);
    Kokkos::View<double***> rhou_new("rhounew", grid.Nx_glob[0]+2*grid.Nghost, 1, 1);
    Kokkos::View<double***> E_new("Enew", grid.Nx_glob[0]+2*grid.Nghost, 1, 1);

    ShockTubeInit(rho, u, P, inter); // Initialisation primary variables (rho, u, P)
    
    ConvPrimConsArray(rho, rhou, E, u, P, eos); // Initialisation conservative variables (rho, rhou, E)
    
    Kokkos::deep_copy(rho_host, rho);
    Kokkos::deep_copy(u_host, u);
    Kokkos::deep_copy(P_host, P);
    Kokkos::deep_copy(rhou_host, rhou);
    Kokkos::deep_copy(E_host, E);
    
    double t = 0;
    int iter = 0;
    bool should_exit = false;

    while (!should_exit && t < timeout && iter<=max_iter)
    {

        face_reconstruction->execute(rho, rhoL, rhoR); // Calcul des pentes
        face_reconstruction->execute(u, uL, uR);
        face_reconstruction->execute(P, PL, PR);
        
        double dt = time_step(cfl, rho, u, P, dx, dx, dx, eos);
        if ((t + dt) > timeout)
        {
            dt = timeout - t;
            should_exit = true;
        }

        ConvPrimConsArray(rhoL, rhouL, EL, uL, PL, eos); // Conversion en variables conservatives
        ConvPrimConsArray(rhoR, rhouR, ER, uR, PR, eos);

        double dto2dx = dt / (2 * dx);

        Kokkos::parallel_for(
            "extrapolation",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {1, 0, 0},
            {grid.Nx_glob[0]+2*grid.Nghost-1, 1, 1}),
            KOKKOS_LAMBDA(int i, int j, int k)
        {
            Flux fluxL(rhoL(i, j, k), uL(i, j, k), PL(i, j, k), eos);
            Flux fluxR(rhoR(i, j, k), uR(i, j, k), PR(i, j, k), eos);  

            rho_moyL(i, j, k) = rhoL(i, j, k) + dto2dx * (fluxL.FluxRho() - fluxR.FluxRho());
            rhou_moyL(i, j, k) = rhouL(i, j, k) + dto2dx * (fluxL.FluxRhou() - fluxR.FluxRhou());
            E_moyL(i, j, k) = EL(i, j, k) + dto2dx * (fluxL.FluxE() - fluxR.FluxE());
            rho_moyR(i, j, k) = rhoR(i, j, k) + dto2dx * (fluxL.FluxRho() - fluxR.FluxRho());
            rhou_moyR(i, j, k) = rhouR(i, j, k) + dto2dx * (fluxL.FluxRhou() - fluxR.FluxRhou());
            E_moyR(i, j, k) = ER(i, j, k) + dto2dx * (fluxL.FluxE() - fluxR.FluxE());
        });

        double dtodx = dt / dx;

        Kokkos::parallel_for(
            "new_values",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {grid.Nghost, 0, 0},
            {grid.Nx_glob[0]+grid.Nghost, 1, 1}),
            KOKKOS_LAMBDA(int i, int j, int k)
        {
            SolverHLL FluxM1(rho_moyR(i-1, j, k), rhou_moyR(i-1, j, k), E_moyR(i-1, j, k), rho_moyL(i, j, k), rhou_moyL(i, j, k), E_moyL(i, j, k), eos);
            SolverHLL FluxP1(rho_moyR(i, j, k), rhou_moyR(i, j, k), E_moyR(i, j, k),rho_moyL(i+1, j, k), rhou_moyL(i+1, j, k), E_moyL(i+1, j, k), eos);

            rho_new(i, j, k) = rho(i, j, k) + dtodx * (FluxM1.FinterRho() - FluxP1.FinterRho());
            rhou_new(i, j, k) = rhou(i, j, k) + dtodx * (FluxM1.FinterRhou() -  FluxP1.FinterRhou());
            E_new(i, j, k) = E(i, j, k) + dtodx * (FluxM1.FinterE() -  FluxP1.FinterE());
        });

       GradientNull(rho_new, rhou_new, E_new);

        ConvConsPrimArray(rho_new, rhou_new, E_new, u, P, eos); //Conversion des variables conservatives en primitives
        Kokkos::deep_copy(rho, rho_new);
        Kokkos::deep_copy(rhou, rhou_new);
        Kokkos::deep_copy(E, E_new);

        bool make_output = should_output(iter, output_frequency, max_iter, t, dt, timeout);

        t = t + dt;
        iter++;

        if(make_output)
        {
            write(iter, grid.Nx_glob[0], t, rho.data(), u.data());
        }
    }

    std::printf("Final time = %f and number of iterations = %d  \n", t, iter);

    Kokkos::parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {0, 0, 0},
            {grid.Nx_glob[0]+2*grid.Nghost, 1, 1}),
            KOKKOS_LAMBDA(int i, int j, int k)
        {
            std::printf("%f %f %f \n", rho(i, j, k), u(i, j, k), P(i, j, k));
        });
    
    PDI_finalize();
    PC_tree_destroy(&conf);

    std::printf("%s\n", "---Fin du programme---");
    return 0;
}
