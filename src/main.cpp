/**
 * @file main.cpp
 * Code d'hydrodynamique radiative en devenir...
 */

#include <iostream>
#include <memory>

#include <inih/INIReader.hpp>

#include "ndim.hpp"
#include "eos.hpp"
#include "PerfectGas.hpp"
#include "RadGas.hpp"
#include "array_conversion.hpp"
#include "initialization_interface.hpp"
#include "face_reconstruction.hpp"
#include "time_step.hpp"
#include "grid.hpp"
#include "boundary.hpp"
#include "boundary_distribute.hpp"
#include "euler_equations.hpp"
#include "hydro_reconstruction.hpp"
#include "godunov_scheme.hpp"
#include "mpi_scope_guard.hpp"
#include <pdi.h>
#include "range.hpp"
#include "kronecker.hpp"
#include "kokkos_shortcut.hpp"
#include "io/config.yaml.hpp"
#include "io/io.hpp"
#include "extrapolation_time.hpp"
#include "nova_params.hpp"
#include "factories.hpp"
#include "setup.hpp"
#include "gravity.hpp"

#define GRAVITY

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

    Kokkos::print_configuration(std::cout);

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
    ParamSetup const param_setup(reader);
    Grid const grid(param);
    grid.print_grid();

    print_info("PROBLEM", param.problem);
    print_info("EOS", eos_choice);
    //print_info("GEOMETRIE", geom_choice);

    EOS const eos(param.gamma, param.mu);

    #ifdef UNIFORM
        UniformGravity g(make_uniform_gravity(param));
        using Gravity = UniformGravity;
        print_info("GRAVITY", "UNIFORM");
    #elif defined(POINT_MASS)
        PointMassGravity g(make_point_mass_gravity(param, grid));
        using Gravity = PointMassGravity;
        print_info("GRAVITY", "POINT_MASS");
    #else
        static_assert(false, "Gravity not defined");
    #endif
    
    write_pdi_init(param.max_iter, param.output_frequency, grid, param);

    std::string bc_choice_dir;
    std::array<std::string, ndim*2> bc_choice_faces;
    for(int idim = 0; idim < ndim; idim++)
    {
        bc_choice_dir = reader.Get("Boundary Condition", "BC" + bc_dir[idim], param.bc_choice);
        for (int iface = 0; iface < 2; iface++)
        {
            bc_choice_faces[idim * 2 + iface] = reader.Get("Boundary Condition",
                                                       "BC" + bc_dir[idim]+bc_face[iface],
                                                       bc_choice_dir);
            if(bc_choice_faces[idim * 2 + iface].empty() )
            {
                throw std::runtime_error("boundary condition not fully defined for dimension "
                                         + bc_dir[idim]);
            }
        }
    }

    std::array<std::unique_ptr<IBoundaryCondition>, ndim * 2> bcs_array;
    for(int idim = 0; idim < ndim; idim++)
    {
        for(int iface = 0; iface < 2; iface++)
        {
            bcs_array[idim * 2 + iface] = factory_boundary_construction<Gravity>(
                    bc_choice_faces[idim * 2 + iface], idim, iface, eos, grid, param_setup, Gravity(g));
        }
    }

    DistributedBoundaryCondition const bcs(reader, grid, param, std::move(bcs_array));

    std::unique_ptr<IInitializationProblem> initialization
            = std::make_unique<InitializationSetup<Gravity>>(eos, grid, param_setup, Gravity(g));

    std::unique_ptr<IFaceReconstruction> face_reconstruction
            = factory_face_reconstruction(param.reconstruction_type, grid);

    std::unique_ptr<IExtrapolationReconstruction> time_reconstruction
            = std::make_unique<ExtrapolationTimeReconstruction<Gravity>>(eos, grid, Gravity(g));

    std::unique_ptr<IGodunovScheme> godunov_scheme
            = factory_godunov_scheme<Gravity>(param.riemann_solver, eos, grid, Gravity(g));

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

    KDV_double_4d fx("fx",        grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], param.nfx);
    KV_double_4d fx_new("fx_new", grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], param.nfx);
    KV_double_6d fx_rec("fx_rec", grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, ndim, param.nfx);

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
        initialization->execute(grid.range.no_ghosts(), rho.d_view, u.d_view, P.d_view, fx.d_view);
    }
    conv_prim_to_cons(grid.range.no_ghosts(), rhou.d_view, E.d_view, rho.d_view, u.d_view, P.d_view, eos);

    bcs.execute(rho.d_view, rhou.d_view, E.d_view, fx.d_view);
    
    conv_cons_to_prim(grid.range.all_ghosts(), u.d_view, P.d_view, rho.d_view, rhou.d_view, E.d_view, eos);

    rho.modify_device();
    u.modify_device();
    P.modify_device();
    fx.modify_device();
    rhou.modify_device();
    E.modify_device();

    std::unique_ptr<IHydroReconstruction> reconstruction 
        = std::make_unique<MUSCLHancockHydroReconstruction>(std::move(face_reconstruction), 
                                                            std::move(time_reconstruction), 
                                                            eos, P_rec, u_rec);

    std::vector<std::pair<int, double>> outputs_record;
    if (param.output_frequency > 0)
    {
        outputs_record.emplace_back(iter, t);
        writeXML(grid, outputs_record, grid.x_glob, grid.y_glob, grid.z_glob);
        write_pdi(iter, t, eos.adiabatic_index(), rho, u, P, E, grid.x, grid.y, grid.z, fx);
    }

    std::chrono::steady_clock::time_point const start = std::chrono::steady_clock::now();

    while (!should_exit && t < param.timeout && iter < param.max_iter)
    {
        double dt = time_step(grid.range.all_ghosts(), param.cfl, rho.d_view, u.d_view, P.d_view, eos, grid);
        bool const make_output = should_output(iter, param.output_frequency, param.max_iter, t, dt, param.timeout);
        if ((t + dt) > param.timeout)
        {
            dt = param.timeout - t;
            should_exit = true;
        }
        
        reconstruction->execute(grid.range.with_ghosts(1), dt/2, rho_rec, rhou_rec, E_rec, fx_rec, 
                                rho.d_view, u.d_view, P.d_view, fx.d_view);

        godunov_scheme->execute(grid.range.no_ghosts(), dt, rho.d_view, rhou.d_view, E.d_view, fx.d_view, 
                                rho_rec, rhou_rec, E_rec, fx_rec,
                                rho_new, rhou_new, E_new, fx_new);

        bcs.execute(rho_new, rhou_new, E_new, fx_new);

        conv_cons_to_prim(grid.range.all_ghosts(), u.d_view, P.d_view, rho_new, rhou_new, E_new, eos);
        Kokkos::deep_copy(rho.d_view, rho_new);
        Kokkos::deep_copy(rhou.d_view, rhou_new);
        Kokkos::deep_copy(E.d_view, E_new);
        Kokkos::deep_copy(fx.d_view, fx_new);

        rho.modify_device();
        u.modify_device();
        P.modify_device();
        E.modify_device();
        rhou.modify_device();
        fx.modify_device();

        t += dt;
        iter++;

        if(make_output)
        {
            outputs_record.emplace_back(iter, t);
            writeXML(grid, outputs_record, grid.x_glob, grid.y_glob, grid.z_glob);
            write_pdi(iter, t, eos.adiabatic_index(), rho, u, P, E, grid.x, grid.y, grid.z, fx);
        }
    }

    std::chrono::steady_clock::time_point const end = std::chrono::steady_clock::now();

    if (grid.mpi_rank == 0)
    {
        float const nb_cells = grid.Nx_glob_ng[0] * grid.Nx_glob_ng[1] * grid.Nx_glob_ng[2];
        float const nb_cells_updated = iter * nb_cells;
        float const duration = std::chrono::duration<float>(end - start).count();
        float const perf = nb_cells_updated / duration;
        std::printf("Final time = %f and number of iterations = %d  \n", t, iter);
        std::printf("Mean performance: %f Mcell-updates/s\n", perf * 1e-6F);
        std::printf("--- End ---\n");
    }
    MPI_Comm_free(&(const_cast<Grid&>(grid).comm_cart));
    PDI_finalize();
    PC_tree_destroy(&conf);

    return 0;
}
