/**
 * @file main.cpp
 * Code d'hydrodynamique radiative en devenir...
 */

#include <mpi.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <inih/INIReader.hpp>

#include <PerfectGas.hpp>
#include <RadGas.hpp>
#include <array_conversion.hpp>
#include <config.yaml.hpp>
#include <conservation.hpp>
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
#include <internal_energy.hpp>
#include <io.hpp>
#include <kokkos_shortcut.hpp>
#include <kronecker.hpp>
#include <ndim.hpp>
#include <nova_params.hpp>
#include <paraconf.h>
#include <pdi.h>
#include <pressure_fix.hpp>
#include <range.hpp>
#include <temperature.hpp>
#include <time_step.hpp>
#include <user_step_factory.hpp>

#include "boundary.hpp"
#include "boundary_distribute.hpp"
#include "boundary_factory.hpp"
#include "initialization_interface.hpp"
#include "mpi_scope_guard.hpp"
#include "setup.hpp"

using namespace novapp;

namespace
{

void display_help_message(std::filesystem::path const& executable)
{
    std::cout << "usage: " << executable.filename().native() << " <path to the ini file> [options]\n";
}

}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        display_help_message(argv[0]);
        return EXIT_FAILURE;
    }

    MpiScopeGuard const mpi_guard(argc, argv);

    Kokkos::ScopeGuard const guard(argc, argv);

    Kokkos::print_configuration(std::cout);

    INIReader const reader(argv[1]);

    std::filesystem::path io_config_path;
    for(int iarg = 2; iarg < argc; ++iarg)
    {
        std::string_view const option(argv[iarg]);
        std::string_view const prefix("--io-config=");
        if (std::string_view::size_type const pos = option.find_first_of(prefix);
            pos != std::string_view::npos)
        {
            io_config_path = option.substr(pos);
        }
    }

    PC_tree_t conf;
    if (!io_config_path.empty())
    {
        conf = PC_parse_path(io_config_path.c_str());
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

     if(grid.mpi_rank==0)
    {
        print_info("SETUP", MY_SETUP);
        print_info("EOS", eos_choice);
        print_info("GEOMETRIE", geom_choice);
    }

#if defined(NOVAPP_GRAVITY_Uniform)
    using Gravity = UniformGravity;
    if(grid.mpi_rank==0)
    {
        print_info("GRAVITY", "Uniform");
    }
#elif defined(NOVAPP_GRAVITY_Point_mass)
    using Gravity = PointMassGravity;
    if(grid.mpi_rank==0)
    {
        print_info("GRAVITY", "Point_mass");
    }
#else
    static_assert(false, "Gravity not defined");
#endif
    std::unique_ptr<Gravity> g;

    std::unique_ptr<IUserStep> user_step;
    if (param.user_step == "UserDefined")
    {
        user_step = std::make_unique<UserStep>();
    }
    else
    {
        user_step = factory_user_step(param.user_step);
    }
    if(grid.mpi_rank==0)
    {
        if (param.pressure_fix == "On")
        {
            print_info("PRESSURE_FIX", param.pressure_fix);
        }
        print_info("RIEMANN_SOLVER", param.riemann_solver);
        print_info("USER_STEP", param.user_step);
    }

    if(param.restart)
    {
        read_pdi(param.restart_file, iter, t, rho, u, P, fx, x_glob, y_glob, z_glob); // read data into host view
        sync_device(x_glob, y_glob, z_glob);
        grid.set_grid(x_glob.d_view, y_glob.d_view, z_glob.d_view);
#if defined(NOVAPP_GRAVITY_Uniform)
        g = std::make_unique<Gravity>(make_uniform_gravity(param));
#elif defined(NOVAPP_GRAVITY_Point_mass)
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
        grid_type->execute(grid.Nghost, grid.Nx_glob_ng, x_glob.h_view, y_glob.h_view, z_glob.h_view);
        modify_host(x_glob, y_glob, z_glob);
        sync_device(x_glob, y_glob, z_glob);
        grid.set_grid(x_glob.d_view, y_glob.d_view, z_glob.d_view);
#if defined(NOVAPP_GRAVITY_Uniform)
        g = std::make_unique<Gravity>(make_uniform_gravity(param));
#elif defined(NOVAPP_GRAVITY_Point_mass)
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

    conv_prim_to_cons(grid.range.no_ghosts(), eos, rho.d_view, u.d_view, P.d_view, rhou.d_view, E.d_view);

    bcs(rho.d_view, rhou.d_view, E.d_view, fx.d_view);

    conv_cons_to_prim(grid.range.all_ghosts(), eos, rho.d_view, rhou.d_view, E.d_view, u.d_view, P.d_view);

    modify_device(rho, u, P, fx, rhou, E);

    double initial_mass = conservation(grid.range.no_ghosts(), grid, rho.d_view);

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
        double dt = time_step(grid.range.all_ghosts(), eos, grid, param.cfl, rho.d_view, u.d_view, P.d_view);
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

        double min_internal_energy = internal_energy(grid.range.no_ghosts(), grid, rho.d_view, rhou.d_view, E.d_view);
        if (min_internal_energy < 0)
        {
            throw std::runtime_error("Internal energy < 0");
        }

        reconstruction->execute(grid.range.with_ghosts(1), dt/2, rho.d_view, u.d_view, P.d_view, fx.d_view,
                                rho_rec, rhou_rec, E_rec, fx_rec);

        godunov_scheme->execute(grid.range.no_ghosts(), dt, rho.d_view, rhou.d_view, E.d_view, fx.d_view,
                                rho_rec, rhou_rec, E_rec, fx_rec,
                                rho_new, rhou_new, E_new, fx_new);

        if (param.pressure_fix == "On")
        {
            pressure_fix(grid.range.no_ghosts(), eos, grid, dt, param.eps_pf,
                        rho.d_view, rhou.d_view, E.d_view,
                        rho_rec, rhou_rec, E_rec,
                        rho_new, rhou_new, E_new);
        }

        user_step->execute(grid.range.no_ghosts(), t, dt, rho_new, E_new, fx_new);

        bcs(rho_new, rhou_new, E_new, fx_new);

        conv_cons_to_prim(grid.range.all_ghosts(), eos, rho_new, rhou_new, E_new, u.d_view, P.d_view);

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

    double final_mass = conservation(grid.range.no_ghosts(), grid, rho.d_view);

    Kokkos::fence();
    MPI_Barrier(grid.comm_cart);

    std::chrono::steady_clock::time_point const end = std::chrono::steady_clock::now();

    if (grid.mpi_rank == 0)
    {
        double const nx_glob_ng = grid.Nx_glob_ng[0];
        double const ny_glob_ng = grid.Nx_glob_ng[1];
        double const nz_glob_ng = grid.Nx_glob_ng[2];
        double const nb_cells = nx_glob_ng * ny_glob_ng * nz_glob_ng;
        double const nb_iter = iter;
        double const duration = std::chrono::duration<double>(end - start).count();
        double const nb_cell_updates_per_sec = nb_iter * nb_cells / duration;
        double const mega = 1E-6;
        double mass_change = std::abs(initial_mass - final_mass);
        std::printf("Final time = %f and number of iterations = %d\n", t, iter);
        std::printf("Mean performance: %f Mcell-updates/s\n", mega * nb_cell_updates_per_sec);
        std::printf("Initial mass = %f and change in mass = %.10e\n", initial_mass, mass_change);
        std::printf("--- End ---\n");
    }
    MPI_Comm_free(&(const_cast<Grid&>(grid).comm_cart));
    PDI_finalize();
    PC_tree_destroy(&conf);

    return 0;
}
