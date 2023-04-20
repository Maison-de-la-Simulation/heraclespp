/**
 * @file main.cpp
 * Code d'hydrodynamique radiative en devenir...
 */

#include <iostream>
#include <memory>

#include <inih/INIReader.hpp>

#include "array_conversion.hpp"
#include "initialisation_problem.hpp"
#include "io.hpp"
#include "face_reconstruction.hpp"
#include "cfl_cond.hpp"
#include "grid.hpp"
#include "boundary.hpp"
#include "euler_equations.hpp"
#include "hydro_reconstruction.hpp"
#include "PerfectGas.hpp"
#include "godunov_scheme.hpp"
#include "mpi_scope_guard.hpp"
#include "ndim.hpp"
#include <pdi.h>
#include "range.hpp"
#include "kronecker.hpp"
#include "Kokkos_shortcut.hpp"
#include "io_config.yaml.hpp"
#include "extrapolation_time.hpp"
#include "nova_params.hpp"

using namespace novapp;

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cout << "usage: " << argv[0] << " <path to the ini file> [<path to the yaml file>]\n";
        return EXIT_FAILURE;
    }

    MpiScopeGuard const mpi_guard;

    Kokkos::ScopeGuard const guard;

    INIReader const reader(argv[1]);

    PC_tree_t conf;
    if (argc == 3)
    {
        conf = PC_parse_path(argv[2]);
    }
    else
    {
        conf = PC_parse_string(io_config);
    }
    PDI_init(PC_get(conf, ".pdi"));

    Param const param(reader);
    Grid const grid(param);
    grid.print_grid();

    thermodynamics::PerfectGas const eos(param.gamma, param.mu);

    KV_double_1d g_array("g_array", 3);
    Kokkos::parallel_for(1, KOKKOS_LAMBDA([[maybe_unused]] int i)
    {
        g_array(0) = param.gx;
        g_array(1) = param.gy;
        g_array(2) = param.gz;
    });

    write_pdi_init(param.max_iter, param.output_frequency, grid);

    DistributedBoundaryCondition const bcs(reader, param, grid);

    std::unique_ptr<IInitialisationProblem> initialisation
            = factory_initialisation(param.initialisation_problem);

    std::unique_ptr<IFaceReconstruction> face_reconstruction
            = factory_face_reconstruction(param.reconstruction_type);
    
    std::unique_ptr<IExtrapolationReconstruction> time_reconstruction
            = factory_time_reconstruction(param.gravity_type, eos, g_array);

    std::unique_ptr<IGodunovScheme> godunov_scheme
            = factory_godunov_scheme(param.riemann_solver, param.gravity_type, eos, g_array);

    KDV_double_3d rho("rho",   grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2]); // Density
    KDV_double_4d u("u",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], ndim); // Velocity
    KDV_double_3d P("P",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2]); // Pressure
    KDV_double_4d rhou("rhou", grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], ndim); // Momentum
    KDV_double_3d E("E",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2]); // Energy

    KV_double_5d rho_rec("rho_rec",   grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, ndim);
    KV_double_6d rhou_rec("rhou_rec", grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, ndim, ndim);
    KV_double_5d E_rec("E_rec",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, ndim);
    KV_double_6d u_rec("u_rec",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, ndim, ndim);
    KV_double_5d P_rec("P_rec",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, ndim);

    KV_double_3d rho_new("rhonew",   grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2]);
    KV_double_4d rhou_new("rhounew", grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], ndim);
    KV_double_3d E_new("Enew",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2]);

    double t = 0;
    int iter = 0;
    bool should_exit = false;

    if(param.restart)
    {   
        read_pdi(param.restart_file, rho, u, P, t, iter); // read data into host view
        
        rho.sync_device();
        u.sync_device();
        P.sync_device();
        
        if(grid.mpi_rank==0) 
        {
            std::cout<<std::endl<< std::left << std::setw(80) << std::setfill('*') << "*"<<std::endl;
            std::cout<<"read from file "<<param.restart_file<<std::endl;
            std::cout<<"starting at time "<<t<<" ( ~ "<<100*t/param.timeout<<"%)"
                     <<", with iteration "<<iter<<std::endl<<std::endl;
        }
    }
    else
    {
        initialisation->execute(grid.range.no_ghosts(), rho.d_view, u.d_view, P.d_view, g_array, eos, grid, param);
    }
    conv_prim_to_cons(grid.range.no_ghosts(), rhou.d_view, E.d_view, rho.d_view, u.d_view, P.d_view, eos);

    bcs.execute(rho.d_view, rhou.d_view, E.d_view, g_array, grid, eos, param);
    
    conv_cons_to_prim(grid.range.all_ghosts(), u.d_view, P.d_view, rho.d_view, rhou.d_view, E.d_view, eos);

    std::unique_ptr<IHydroReconstruction> reconstruction = std::make_unique<
            MUSCLHancockHydroReconstruction>(std::move(face_reconstruction), std::move(time_reconstruction), P_rec, u_rec);

    if (param.output_frequency > 0)
    {
        write_pdi(iter, t, eos.adiabatic_index(), rho, u, P, E, grid.x, grid.y, grid.z);
    }

    while (!should_exit && t < param.timeout && iter < param.max_iter)
    {
        double dt = time_step(grid.range.all_ghosts(), param.cfl, rho.d_view, u.d_view, P.d_view, eos, grid);
        bool const make_output = should_output(iter, param.output_frequency, param.max_iter, t, dt, param.timeout);
        if ((t + dt) > param.timeout)
        {
            dt = param.timeout - t;
            should_exit = true;
        }

        reconstruction->execute(grid.range.with_ghosts(1), rho_rec, rhou_rec, E_rec, rho.d_view, u.d_view, P.d_view, 
                                eos, dt/2, grid);

        godunov_scheme->execute(grid.range.no_ghosts(), rho.d_view, rhou.d_view, E.d_view, rho_rec, rhou_rec, E_rec,
                                rho_new, rhou_new, E_new, dt, grid);

        bcs.execute(rho_new, rhou_new, E_new, g_array, grid, eos, param);

        conv_cons_to_prim(grid.range.all_ghosts(), u.d_view, P.d_view, rho_new, rhou_new, E_new, eos);
        Kokkos::deep_copy(rho.d_view, rho_new);
        Kokkos::deep_copy(rhou.d_view, rhou_new);
        Kokkos::deep_copy(E.d_view, E_new);

        rho.modify_device();
        u.modify_device();
        P.modify_device();
        E.modify_device();
        rhou.modify_device();

        t += dt;
        iter++;

        if(make_output)
        {
            write_pdi(iter, t, eos.adiabatic_index(), rho, u, P, E, grid.x, grid.y, grid.z);
        }
    }

    if (grid.mpi_rank == 0)
    {
        std::printf("Final time = %f and number of iterations = %d  \n", t, iter);
        std::printf("--- End ---\n");
    }
    MPI_Comm_free(&(const_cast<Grid&>(grid).comm_cart));
    PDI_finalize();
    PC_tree_destroy(&conf);

    return 0;
}
