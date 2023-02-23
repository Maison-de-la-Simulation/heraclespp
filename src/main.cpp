/**
 * @file main.cpp
 * Code d'hydrodynamique radiative en devenir...
 */

#include <iostream>
#include <memory>

#include <inih/INIReader.hpp>

#include <Kokkos_Core.hpp>

#include "array_conversion.hpp"
#include "initialisation_problem.hpp"
#include "io.hpp"
#include "face_reconstruction.hpp"
#include "cfl_cond.hpp"
#include "grid.hpp"
#include "boundary.hpp"
#include "euler_equations.hpp"
#include "extrapolation_construction.hpp"
#include "PerfectGas.hpp"
#include "godunov_scheme.hpp"
#include "mpi_scope_guard.hpp"
#include <pdi.h>

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cout << "usage: " << argv[0] << " <path to the ini file> <path to the yaml file>\n";
        return EXIT_FAILURE;
    }

    MpiScopeGuard const mpi_guard;

    Kokkos::ScopeGuard const guard;

    INIReader const reader(argv[1]);

    PC_tree_t conf = PC_parse_path(argv[2]);
    PDI_init(PC_get(conf, ".pdi"));

    Grid const grid(reader);
    grid.print_grid();

    double const timeout = reader.GetReal("Run", "timeout", 0.2);
    double const cfl = reader.GetReal("Run", "cfl", 0.4);

    int const max_iter = reader.GetInteger("Output", "max_iter", 10000);
    int const output_frequency = reader.GetInteger("Output", "frequency", 10);

    thermodynamics::PerfectGas const eos(reader.GetReal("PerfectGas", "gamma", 1.4), 1.0);

    Kokkos::View<double*> array_dx("array_dx", 3); //Space step array
    Kokkos::parallel_for(3, KOKKOS_LAMBDA(int i)
    {
        array_dx(i) = 1. / grid.Nx_glob_ng[i];
    });

    write_pdi_init(max_iter, output_frequency, grid);
    

    std::string const initialisation_problem = reader.Get("Problem", "type", "ShockTube");
    std::unique_ptr<IInitialisationProblem> initialisation
            = factory_initialisation(initialisation_problem);

    std::string const reconstruction_type = reader.Get("Hydro", "reconstruction", "Minmod");
    std::unique_ptr<IFaceReconstruction> face_reconstruction
            = factory_face_reconstruction(reconstruction_type);

    std::unique_ptr<IExtrapolationValues> extrapolation_construction
            = std::make_unique<ExtrapolationCalculation>();

    std::string const boundary_condition_type = reader.Get("Hydro", "boundary", "NullGradient");
    std::unique_ptr<IBoundaryCondition> boundary_construction
            = factory_boundary_construction(grid, boundary_condition_type);
    
    std::string const riemann_solver = reader.Get("Hydro", "riemann_solver", "HLL");
    std::unique_ptr<IGodunovScheme> godunov_scheme
            = factory_godunov_scheme(riemann_solver, eos);

    Kokkos::View<double*> nodes_x0("nodes_x0", grid.Nx_local_wg[0]+1); // Nodes for x0

    int offset = grid.range.Corner_min[0] - grid.Nghost[0];
    Kokkos::parallel_for("InitialisationNodes",
                         Kokkos::RangePolicy<>(0, nodes_x0.extent(0)),
                         KOKKOS_LAMBDA(int i)
    {
        nodes_x0(i) = (i +offset) * array_dx(0) ; // Position of the left interface
    });

    Kokkos::View<double***>  rho("rho",   grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2]); // Density
    Kokkos::View<double****> rhou("rhou", grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], ndim); // Momentum
    Kokkos::View<double***>  E("E",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2]); // Energy
    Kokkos::View<double****> u("u",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], ndim); // Speed
    Kokkos::View<double***>  P("P",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2]); // Pressure
    
    Kokkos::View<double***> ::HostMirror rho_host
                            = Kokkos::create_mirror_view(rho); // Density always on host
    Kokkos::View<double****>::HostMirror rhou_host
                            = Kokkos::create_mirror_view(rhou); // Momentum always on host
    Kokkos::View<double***> ::HostMirror E_host
                            = Kokkos::create_mirror_view(E); // Energy always on host
    Kokkos::View<double****>::HostMirror u_host
                            = Kokkos::create_mirror_view(u); // Speedalways on host
    Kokkos::View<double***> ::HostMirror P_host
                            = Kokkos::create_mirror_view(P); // Pressure always on host

    Kokkos::View<double*****>  rho_rec("rho_rec",   grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, ndim);
    Kokkos::View<double******> rhou_rec("rhou_rec", grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, ndim, ndim);
    Kokkos::View<double*****>  E_rec("E_rec",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, ndim);
    Kokkos::View<double******> u_rec("u_rec",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, ndim, ndim);
    Kokkos::View<double*****>  P_rec("P_rec",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, ndim);

    Kokkos::View<double***>  rho_new("rhonew",   grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2]);
    Kokkos::View<double****> rhou_new("rhounew", grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], ndim);
    Kokkos::View<double***>  E_new("Enew",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2]);

    initialisation->execute(rho, u, P, nodes_x0);

    ConvPrimtoConsArray(rhou, E, rho, u, P, eos);

    Kokkos::deep_copy(rho_host, rho);
    Kokkos::deep_copy(u_host, u);
    Kokkos::deep_copy(P_host, P);
    Kokkos::deep_copy(rhou_host, rhou);
    Kokkos::deep_copy(E_host, E);

    double t = 0;
    int iter = 0;
    bool should_exit = false; 

    write_pdi(iter, t, rho_host.data(), u_host.data(), P_host.data());

    while (!should_exit && t < timeout && iter < max_iter)
    {
        double dt = time_step(cfl, rho, u, P, array_dx, eos);
        bool const make_output = should_output(iter, output_frequency, max_iter, t, dt, timeout);

        if ((t + dt) > timeout)
        {
            dt = timeout - t;
            should_exit = true;
        }

        face_reconstruction->execute(rho, rho_rec, array_dx);
        face_reconstruction->execute(P, P_rec, array_dx);
        for(int idim = 0; idim < ndim ; ++idim)
        {
            auto u_less_dim = Kokkos::subview(u, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, idim);
            auto u_rec_less_dim = Kokkos::subview(u_rec, Kokkos::ALL, Kokkos::ALL, 
                                    Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, idim);
            face_reconstruction->execute(u_less_dim, u_rec_less_dim, array_dx);
        }

        for(int idim = 0; idim < ndim ; ++idim)
        {
            for (int n = 0; n < 2; ++n)
            {
                auto rhou_rec_less_dim = Kokkos::subview(rhou_rec, Kokkos::ALL, Kokkos::ALL, 
                                                    Kokkos::ALL, n, idim, Kokkos::ALL);
                auto E_rec_less_dim = Kokkos::subview(E_rec, Kokkos::ALL, Kokkos::ALL, 
                                                    Kokkos::ALL, n, idim);
                auto rho_rec_less_dim = Kokkos::subview(rho_rec, Kokkos::ALL, Kokkos::ALL, 
                                                    Kokkos::ALL, n, idim);
                auto u_rec_less_dim = Kokkos::subview(u_rec, Kokkos::ALL, Kokkos::ALL, 
                                                    Kokkos::ALL, n, idim, Kokkos::ALL);
                auto P_rec_less_dim = Kokkos::subview(P_rec, Kokkos::ALL, Kokkos::ALL, 
                                                    Kokkos::ALL, n, idim);
                ConvPrimtoConsArray(rhou_rec_less_dim, E_rec_less_dim, rho_rec_less_dim, 
                                u_rec_less_dim, P_rec_less_dim, eos);
            }
        }

        extrapolation_construction->execute(rhou_rec, E_rec, rho_rec, u_rec, P_rec,
                                        eos, array_dx, dt);

        godunov_scheme->execute(rho, rhou, E, rho_rec, rhou_rec, E_rec, 
                                rho_new, rhou_new, E_new, array_dx, dt);

        boundary_construction->execute(rho_new, rhou_new, E_new, grid);

        ConvConstoPrimArray(u, P, rho_new, rhou_new, E_new, eos);
        Kokkos::deep_copy(rho, rho_new);
        Kokkos::deep_copy(rhou, rhou_new);
        Kokkos::deep_copy(E, E_new);

        t = t + dt;
        iter++;

        if(make_output)
        {
            Kokkos::deep_copy(rho_host, rho);
            Kokkos::deep_copy(u_host, u);
            Kokkos::deep_copy(P_host, P);
            write_pdi(iter, t, rho_host.data(), u_host.data(), P_host.data());
        }
    }

    std::printf("Final time = %f and number of iterations = %d  \n", t, iter);

    PDI_finalize();
    PC_tree_destroy(&conf);
    std::printf("%s\n", "--- End ---");
    return 0;
}
