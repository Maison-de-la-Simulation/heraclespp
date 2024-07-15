/**
 * @file main.cpp
 * Code d'hydrodynamique radiative en devenir...
 */

#include <mpi.h>

#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
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
#include <eos.hpp>
#include <euler_equations.hpp>
#include <extrapolation_time.hpp>
#include <face_reconstruction.hpp>
#include <geom.hpp>
#include <git_version.hpp>
#include <godunov_scheme.hpp>
#include <gravity.hpp>
#include <grid.hpp>
#include <grid_factory.hpp>
#include <hydro_reconstruction.hpp>
#include <integration.hpp>
#include <internal_energy.hpp>
#include <io.hpp>
#include <kokkos_shortcut.hpp>
#include <kronecker.hpp>
#include <ndim.hpp>
#include <nova_params.hpp>
#include <paraconf.h>
#include <pdi.h>
#include <pressure_fix.hpp>
#include <print_info.hpp>
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

#if defined(NOVAPP_GRAVITY_Uniform)
using Gravity = UniformGravity;
std::string_view gravity_label("Uniform");
#elif defined(NOVAPP_GRAVITY_Point_mass)
using Gravity = PointMassGravity;
std::string_view gravity_label("Point_mass");
#else
static_assert(false, "Gravity not defined");
#endif

Gravity make_gravity(Param const& param, [[maybe_unused]] Grid const& grid)
{
#if defined(NOVAPP_GRAVITY_Uniform)
    return make_uniform_gravity(param);
#elif defined(NOVAPP_GRAVITY_Point_mass)
    return make_point_mass_gravity(param, grid);
#endif
}

std::string display_help_message(std::filesystem::path const& executable)
{
    std::stringstream ss;
    ss << "usage: " << executable.filename() << " <path to the ini file> [options]";
    return ss.str();
}

void novapp_main(int argc, char** argv)
{
    if (argc < 2)
    {
        throw std::runtime_error(display_help_message(argv[0]));
    }

    MpiScopeGuard const mpi_guard(argc, argv);

    Kokkos::ScopeGuard const kokkos_guard(argc, argv);

    INIReader const reader(argv[1]);

    std::filesystem::path io_config_path;
    for(int iarg = 2; iarg < argc; ++iarg)
    {
        std::string_view const arg(argv[iarg]);
        std::string_view const option_name("--io-config=");
        if (arg.find(option_name) == 0)
        {
            std::string_view option_value = arg;
            option_value.remove_prefix(option_name.size());
            io_config_path = option_value;
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

    if (grid.mpi_rank == 0)
    {
        Kokkos::print_configuration(std::cout);

        grid.print_grid(std::cout);

        print_info(std::cout, "setup", MY_SETUP);
        print_info(std::cout, "eos", eos_choice);
        print_info(std::cout, "geometry", geom_choice);
        print_info(std::cout, "gravity", gravity_label);
        print_info(std::cout, "pressure_fix", param.pressure_fix);
        print_info(std::cout, "riemann_solver", param.riemann_solver);
        print_info(std::cout, "user_step", param.user_step);
        print_info(std::cout, "git_branch", git_branch);
        print_info(std::cout, "git_build_string", git_build_string);
        print_info(std::cout, "compile_date", compile_date);
        print_info(std::cout, "compile_time", compile_time);
    }


    EOS const eos(param.gamma, param.mu);

    write_pdi_init(grid, param);

    std::string bc_choice_dir;
    std::array<std::string, ndim*2> bc_choice_faces;
    for(int idim = 0; idim < ndim; ++idim)
    {
        bc_choice_dir = reader.Get("Boundary Condition", "BC" + bc_dir[idim], param.bc_choice);
        for (int iface = 0; iface < 2; ++iface)
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
    KDV_double_4d rhou("rhou", grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], ndim); // Momentum
    KDV_double_3d E("E",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2]); // Energy
    KDV_double_4d fx("fx",     grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], param.nfx);
    KDV_double_4d u("u",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], ndim); // Velocity
    KDV_double_3d P("P",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2]); // Pressure
    KDV_double_3d T("T",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2]); // Temperature

    KV_double_5d const rho_rec("rho_rec",   grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, ndim);
    KV_double_6d const rhou_rec("rhou_rec", grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, ndim, ndim);
    KV_double_5d const E_rec("E_rec",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, ndim);
    KV_double_6d const fx_rec("fx_rec",     grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, ndim, param.nfx);
    KV_double_6d const u_rec("u_rec",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, ndim, ndim);
    KV_double_5d const P_rec("P_rec",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, ndim);

    KV_double_3d const rho_new("rhonew",   grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2]);
    KV_double_4d const rhou_new("rhounew", grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], ndim);
    KV_double_3d const E_new("Enew",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2]);
    KV_double_4d const fx_new("fx_new",    grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], param.nfx);

    int output_id = -1;
    int time_output_id = -1;
    int iter_output_id = -1;
    std::vector<std::pair<int, double>> outputs_record;
    int const iter_ini = 0;
    double t = param.t_ini;
    int iter = iter_ini;
    bool should_exit = false;
    std::chrono::hours const time_save(param.time_job);

    KDV_double_1d x_glob("x_glob", grid.Nx_glob_ng[0]+2*grid.Nghost[0]+1);
    KDV_double_1d y_glob("y_glob", grid.Nx_glob_ng[1]+2*grid.Nghost[1]+1);
    KDV_double_1d z_glob("z_glob", grid.Nx_glob_ng[2]+2*grid.Nghost[2]+1);

    std::unique_ptr<Gravity> g;

    if(param.restart) // complete restart with a file fom the code
    {
        read_pdi(param.restart_file, output_id, iter_output_id, time_output_id, iter, t, rho, u, P, fx, x_glob, y_glob, z_glob); // read data into host view
        sync_device(x_glob, y_glob, z_glob);
        grid.set_grid(x_glob.d_view, y_glob.d_view, z_glob.d_view);

        sync_device(rho, u, P, fx);
        g = std::make_unique<Gravity>(make_gravity(param, grid));

        if(grid.mpi_rank==0)
        {
            std::cout << '\n' << std::left << std::setw(81) << std::setfill('*') << '\n';
            std::cout << "read from file " << param.restart_file << '\n';
            std::cout << "starting at time " << t << " ( ~ "<<100*t/param.t_end<<"%)"
                    << ", with iteration  "<< iter << "\n\n";
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
        g = std::make_unique<Gravity>(make_gravity(param, grid));
        std::unique_ptr<IInitializationProblem> initialization
            = std::make_unique<InitializationSetup<Gravity>>(eos, param_setup, *g);
        initialization->execute(grid.range.no_ghosts(), grid, rho.d_view, u.d_view, P.d_view, fx.d_view);
    }

    // Create the operators of the main time loop
    std::array<std::unique_ptr<IBoundaryCondition<Gravity>>, ndim * 2> bcs_array;
    for(int idim = 0; idim < ndim; ++idim)
    {
        for(int iface = 0; iface < 2; ++iface)
        {
            if (bc_choice_faces[idim * 2 + iface] == "UserDefined")
            {
                bcs_array[idim * 2 + iface] = std::make_unique<BoundarySetup<Gravity>>(idim, iface, eos, param_setup);
            }
            else
            {
                bcs_array[idim * 2 + iface] = factory_boundary_construction<Gravity>(
                    bc_choice_faces[idim * 2 + iface], idim, iface);
            }
        }
    }

    DistributedBoundaryCondition const bcs(grid, param);

    std::unique_ptr<IFaceReconstruction> face_reconstruction
            = factory_face_reconstruction(param.reconstruction_type);

    std::unique_ptr<IExtrapolationReconstruction<Gravity>> time_reconstruction
            = std::make_unique<ExtrapolationTimeReconstruction<EOS, Gravity>>(eos);

    std::unique_ptr<IHydroReconstruction<Gravity>> reconstruction
        = std::make_unique<MUSCLHancockHydroReconstruction<EOS, Gravity>>(std::move(face_reconstruction),
                                                            std::move(time_reconstruction),
                                                            eos, P_rec, u_rec);

    std::unique_ptr<IGodunovScheme<Gravity>> godunov_scheme
            = factory_godunov_scheme<EOS, Gravity>(param.riemann_solver, eos);

    std::unique_ptr<IUserStep> user_step;
    if (param.user_step == "UserDefined")
    {
        user_step = std::make_unique<UserStep>();
    }
    else
    {
        user_step = factory_user_step(param.user_step);
    }

    conv_prim_to_cons(grid.range.no_ghosts(), eos, rho.d_view, u.d_view, P.d_view, rhou.d_view, E.d_view);

    bcs(bcs_array, grid, *g, rho.d_view, rhou.d_view, E.d_view, fx.d_view);

    conv_cons_to_prim(grid.range.all_ghosts(), eos, rho.d_view, rhou.d_view, E.d_view, u.d_view, P.d_view);

    modify_device(rho, u, P, fx, rhou, E);

    XmlWriter const xml_writer(param.directory, param.prefix, param.nfx);

    if (!param.restart && (param.iter_output_frequency > 0 || param.time_output_frequency > 0))
    {
        ++iter_output_id;
        ++time_output_id;

        temperature(grid.range.all_ghosts(), eos, rho.d_view, P.d_view, T.d_view);
        modify_device(T);

        outputs_record.emplace_back(iter, t);
        ++output_id;
        xml_writer(grid, output_id, outputs_record, x_glob, y_glob, z_glob);
        write_pdi(param.directory, param.prefix, output_id, iter_output_id, time_output_id, iter, t, eos.adiabatic_index(), grid, rho, u, P, E, x_glob, y_glob, z_glob, fx, T);
        print_simulation_status(std::cout, iter, t, param.t_end, output_id);
    }

    double const initial_mass = integrate(grid.range.no_ghosts(), grid, rho.d_view);

    Kokkos::fence("Nova++: before main time loop");
    MPI_Barrier(grid.comm_cart);
    std::chrono::steady_clock::time_point const start = std::chrono::steady_clock::now();
    Kokkos::Profiling::pushRegion("Nova++: main time loop");

    while (!should_exit)
    {
        double dt = param.cfl * time_step(grid.range.all_ghosts(), eos, grid, rho.d_view, u.d_view, P.d_view);

        bool make_output = false;
        if (param.iter_output_frequency > 0)
        {
            int const next_output = iter_ini + (iter_output_id + 1) * param.iter_output_frequency;
            if ((iter + 1) >= next_output)
            {
                make_output = true;
                ++iter_output_id;
            }
        }
        if (param.time_output_frequency > 0)
        {
            double const next_output = param.t_ini + (time_output_id + 1) * param.time_output_frequency;
            if ((t + dt) >= next_output)
            {
                dt = next_output - t;
                make_output = true;
                ++time_output_id;
            }
        }

        if ((t + dt) >= param.t_end)
        {
            dt = param.t_end - t;
            make_output = true;
            should_exit = true;
        }
        if ((iter + 1) >= param.max_iter)
        {
            should_exit = true;
        }

        {
            // if time save > duration simulation
            bool save_and_exit = (std::chrono::steady_clock::now() - start) >= time_save;
            MPI_Bcast(&save_and_exit, 1, MPI_CXX_BOOL, 0, grid.comm_cart);
            if (save_and_exit)
            {
                make_output = true;
                should_exit = true;
            }
        }

        double const min_internal_energy = minimum_internal_energy(grid.range.no_ghosts(), grid, rho.d_view, rhou.d_view, E.d_view);
        if (Kokkos::isnan(min_internal_energy) || min_internal_energy < 0)
        {
            std::stringstream ss;
            ss << "Time = " << t << ", iteration = " << iter;
            ss << ": detected invalid volumic internal energy";
            throw std::runtime_error(ss.str());
        }

        reconstruction->execute(grid.range.with_ghosts(1), grid, *g, dt/2,
                                rho.d_view, u.d_view, P.d_view, fx.d_view,
                                rho_rec, rhou_rec, E_rec, fx_rec);

        godunov_scheme->execute(grid.range.no_ghosts(), grid, *g, dt,
                                rho.d_view, rhou.d_view, E.d_view, fx.d_view,
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

        bcs(bcs_array, grid, *g, rho_new, rhou_new, E_new, fx_new);

        conv_cons_to_prim(grid.range.all_ghosts(), eos, rho_new, rhou_new, E_new, u.d_view, P.d_view);

        Kokkos::deep_copy(rho.d_view, rho_new);
        Kokkos::deep_copy(rhou.d_view, rhou_new);
        Kokkos::deep_copy(E.d_view, E_new);
        Kokkos::deep_copy(fx.d_view, fx_new);

        modify_device(rho, u, P, E, rhou, fx);

        t += dt;
        ++iter;

        if(make_output)
        {
            temperature(grid.range.all_ghosts(), eos, rho.d_view, P.d_view, T.d_view);
            modify_device(T);

            outputs_record.emplace_back(iter, t);
            ++output_id;
            xml_writer(grid, output_id, outputs_record, x_glob, y_glob, z_glob);
            write_pdi(param.directory, param.prefix, output_id, iter_output_id, time_output_id, iter, t, eos.adiabatic_index(), grid, rho, u, P, E, x_glob, y_glob, z_glob, fx, T);
            print_simulation_status(std::cout, iter, t, param.t_end, output_id);
        }
    }

    Kokkos::fence("Nova++: after main time loop");
    MPI_Barrier(grid.comm_cart);
    Kokkos::Profiling::popRegion();
    std::chrono::steady_clock::time_point const end = std::chrono::steady_clock::now();

    double const final_mass = integrate(grid.range.no_ghosts(), grid, rho.d_view);

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
        double const mass_change = std::abs(initial_mass - final_mass);
        std::cout << "Final time = " << t << " and number of iterations = " << iter << '\n';
        std::cout << "Mean performance: " << mega * nb_cell_updates_per_sec << " Mcell-updates/s\n";
        std::cout << "Initial mass = " << initial_mass << " and change in mass = " << mass_change << '\n';
        std::cout << "--- End ---\n";
    }

    PDI_finalize();
    PC_tree_destroy(&conf);
}

}

int main(int argc, char** argv)
{
    try
    {
        novapp_main(argc, argv);
    }
    catch(std::exception const& e)
    {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
