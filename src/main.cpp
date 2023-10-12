/**
 * @file main.cpp
 * Code d'hydrodynamique radiative en devenir...
 */

#include <mpi.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <inih/INIReader.hpp>

#include <PerfectGas.hpp>
#include <RadGas.hpp>
#include <array_conversion.hpp>
#include <config.yaml.hpp>
#include <eos.hpp>
#include <euler_equations.hpp>
#include <extrapolation_time.hpp>
#include <face_reconstruction.hpp>
#include <geom.hpp>
#include <godunov_scheme.hpp>
#include <gravity.hpp>
#include <grid.hpp>
#include <grid_factory.hpp>
#include <hydro_reconstruction.hpp>
#include <io.hpp>
#include <kokkos_shortcut.hpp>
#include <kronecker.hpp>
#include <ndim.hpp>
#include <nova_params.hpp>
#include <paraconf.h>
#include <pdi.h>
#include <range.hpp>
#include <temperature.hpp>
#include <time_step.hpp>
#include <user_step.hpp>

#include "boundary.hpp"
#include "boundary_distribute.hpp"
#include "boundary_factory.hpp"
#include "initialization_interface.hpp"
#include "mpi_scope_guard.hpp"
#include "setup.hpp"

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
    Grid grid(param);
    grid.print_grid();

    print_info("SETUP", MY_SETUP);
    print_info("EOS", eos_choice);
    print_info("GEOMETRIE", geom_choice);

    EOS const eos(param.gamma, param.mu);

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

    KDV_double_3d rho("rho",   grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2]); // Density
    KDV_double_4d u("u",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], ndim); // Velocity
    KDV_double_3d P("P",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2]); // Pressure
    KDV_double_4d rhou("rhou", grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], ndim); // Momentum
    KDV_double_3d E("E",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2]); // Energy
    KDV_double_3d T("T",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2]); // Temperature

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

    KDV_double_1d x_glob("x_glob", grid.Nx_glob_ng[0]+2*grid.Nghost[0]+1);
    KDV_double_1d y_glob("y_glob", grid.Nx_glob_ng[1]+2*grid.Nghost[1]+1);
    KDV_double_1d z_glob("z_glob", grid.Nx_glob_ng[2]+2*grid.Nghost[2]+1);

#if defined(Uniform)
    using Gravity = UniformGravity;
    print_info("GRAVITY", "Uniform");
#elif defined(Point_mass)
    using Gravity = PointMassGravity;
    print_info("GRAVITY", "Point_mass");
#else
    static_assert(false, "Gravity not defined");
#endif
    std::unique_ptr<Gravity> g;

    std::unique_ptr<I_User_Step> user_step
        = factory_user_step(param.user_step);

    if (param.user_step == "UserDefined")
    {
        user_step = std::make_unique<User_Step>();
    }
    print_info("USER STEP", param.user_step);

    if(param.restart)
    {   
        read_pdi(param.restart_file, rho, u, P, fx, x_glob, y_glob, z_glob, t, iter); // read data into host view
        sync_device(x_glob, y_glob, z_glob);
        grid.set_grid(x_glob.d_view, y_glob.d_view, z_glob.d_view);
#if defined(Uniform)
        g = std::make_unique<Gravity>(make_uniform_gravity(param));
#elif defined(Point_mass)
        g = std::make_unique<Gravity>(make_point_mass_gravity(param, grid));
#endif

        sync_device(rho, u, P, fx);

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
        std::unique_ptr<IGridType> grid_type;
        if (param.grid_type == "UserDefined")
        {
            grid_type = std::make_unique<GridSetup>(param);
        }
        else
        {
            grid_type = factory_grid_type(param.grid_type, param);
        }
        grid_type->execute(x_glob.h_view, y_glob.h_view, z_glob.h_view, grid.Nghost, grid.Nx_glob_ng);
        modify_host(x_glob, y_glob, z_glob);
        sync_device(x_glob, y_glob, z_glob);
        grid.set_grid(x_glob.d_view, y_glob.d_view, z_glob.d_view);
#if defined(Uniform)
        g = std::make_unique<Gravity>(make_uniform_gravity(param));
#elif defined(Point_mass)
        g = std::make_unique<Gravity>(make_point_mass_gravity(param, grid));
#endif
        std::unique_ptr<IInitializationProblem> initialization
            = std::make_unique<InitializationSetup<Gravity>>(eos, grid, param_setup, *g);
        initialization->execute(grid.range.no_ghosts(), rho.d_view, u.d_view, P.d_view, fx.d_view);
    }

    std::array<std::unique_ptr<IBoundaryCondition>, ndim * 2> bcs_array;
    for(int idim = 0; idim < ndim; idim++)
    {
        for(int iface = 0; iface < 2; iface++)
        {
            if (bc_choice_faces[idim * 2 + iface] == "UserDefined")
            {
                bcs_array[idim * 2 + iface] = std::make_unique<BoundarySetup<Gravity>>(idim, iface, eos, grid, param_setup, *g);
            }
            else
            {
                bcs_array[idim * 2 + iface] = factory_boundary_construction(
                    bc_choice_faces[idim * 2 + iface], idim, iface, grid);
            }
        }
    }

    DistributedBoundaryCondition const bcs(grid, param, std::move(bcs_array));

    std::unique_ptr<IFaceReconstruction> face_reconstruction
            = factory_face_reconstruction(param.reconstruction_type, grid);

    std::unique_ptr<IExtrapolationReconstruction> time_reconstruction
            = std::make_unique<ExtrapolationTimeReconstruction<Gravity>>(eos, grid, *g);

    std::unique_ptr<IHydroReconstruction> reconstruction 
        = std::make_unique<MUSCLHancockHydroReconstruction>(std::move(face_reconstruction), 
                                                            std::move(time_reconstruction), 
                                                            eos, P_rec, u_rec);

    std::unique_ptr<IGodunovScheme> godunov_scheme
            = factory_godunov_scheme(param.riemann_solver, eos, grid, *g);

    conv_prim_to_cons(grid.range.no_ghosts(), rhou.d_view, E.d_view, rho.d_view, u.d_view, P.d_view, eos);

    bcs(rho.d_view, rhou.d_view, E.d_view, fx.d_view);

    conv_cons_to_prim(grid.range.all_ghosts(), u.d_view, P.d_view, rho.d_view, rhou.d_view, E.d_view, eos);

    modify_device(rho, u, P, fx, rhou, E);

    std::vector<std::pair<int, double>> outputs_record;
    if (param.output_frequency > 0)
    {
        temperature(grid.range.all_ghosts(), eos, rho.d_view, P.d_view, T.d_view);
        modify_device(T);

        outputs_record.emplace_back(iter, t);
        writeXML(grid, outputs_record, x_glob, y_glob, z_glob);
        write_pdi(iter, t, eos.adiabatic_index(), rho, u, P, E, x_glob, y_glob, z_glob, fx, T);
    }

    should_output_fn const should_output(param.output_frequency, param.max_iter, param.timeout);

    std::chrono::steady_clock::time_point const start = std::chrono::steady_clock::now();

    while (!should_exit)
    {
        double dt = time_step(grid.range.all_ghosts(), param.cfl, rho.d_view, u.d_view, P.d_view, eos, grid);
        bool const make_output = should_output(iter, t, dt);
        if ((t + dt) > param.timeout)
        {
            dt = param.timeout - t;
            should_exit = true;
        }
        if ((iter + 1) > param.max_iter)
        {
            should_exit = true;
        }

        reconstruction->execute(grid.range.with_ghosts(1), dt/2, rho_rec, rhou_rec, E_rec, fx_rec, 
                                rho.d_view, u.d_view, P.d_view, fx.d_view);

        godunov_scheme->execute(grid.range.no_ghosts(), dt, rho.d_view, rhou.d_view, E.d_view, fx.d_view,
                                rho_rec, rhou_rec, E_rec, fx_rec,
                                rho_new, rhou_new, E_new, fx_new);

        user_step->execute(grid.range.no_ghosts(), t, dt, rho_new, E_new, fx_new);

        bcs(rho_new, rhou_new, E_new, fx_new);

        conv_cons_to_prim(grid.range.all_ghosts(), u.d_view, P.d_view, rho_new, rhou_new, E_new, eos);

        Kokkos::deep_copy(rho.d_view, rho_new);
        Kokkos::deep_copy(rhou.d_view, rhou_new);
        Kokkos::deep_copy(E.d_view, E_new);
        Kokkos::deep_copy(fx.d_view, fx_new);

        modify_device(rho, u, P, E, rhou, fx);

        t += dt;
        iter++;

        if(make_output)
        {
            temperature(grid.range.all_ghosts(), eos, rho.d_view, P.d_view, T.d_view);
            modify_device(T);

            outputs_record.emplace_back(iter, t);
            writeXML(grid, outputs_record, x_glob, y_glob, z_glob);
            write_pdi(iter, t, eos.adiabatic_index(), rho, u, P, E, x_glob, y_glob, z_glob, fx, T);
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
