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
#include "extrapolation_construction.hpp"
#include "PerfectGas.hpp"
#include "godunov_scheme.hpp"
#include "mpi_scope_guard.hpp"
#include "ndim.hpp"
#include <pdi.h>
#include "range.hpp"
#include "kronecker.hpp"
#include "gravity_implementation.hpp"
#include "Kokkos_shortcut.hpp"

using namespace novapp;

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

    double const xmin = reader.GetReal("Grid", "xmin", 0.0);
    double const xmax = reader.GetReal("Grid", "xmax", 1.0);
    double const ymin = reader.GetReal("Grid", "ymin", 0.0);
    double const ymax = reader.GetReal("Grid", "ymax", 1.0);
    double const zmin = reader.GetReal("Grid", "zmin", 0.0);
    double const zmax = reader.GetReal("Grid", "zmax", 1.0);

    double const Lx = xmax - xmin;
    double const Ly = ymax - ymin;
    double const Lz = zmax - zmin;

    KV_double_1d array_dx("array_dx", 3); //Space step array
    Kokkos::parallel_for(1, KOKKOS_LAMBDA(int i)
    {
        array_dx(0) = Lx / grid.Nx_glob_ng[0];
        array_dx(1) = Ly / grid.Nx_glob_ng[1];
        array_dx(2) = Lz / grid.Nx_glob_ng[2];
    });

    write_pdi_init(max_iter, output_frequency, grid);
    
    std::string const initialisation_problem = reader.Get("Problem", "type", "ShockTube");
    std::unique_ptr<IInitialisationProblem> initialisation
            = factory_initialisation(initialisation_problem);

    std::string const reconstruction_type = reader.Get("Hydro", "reconstruction", "VanLeer");
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

    std::string const gravity = reader.Get("Gravity", "gravity", "Off");
    std::unique_ptr<IGravity> gravity_add
            = factory_gravity_source(gravity);

    double const g = reader.GetReal("Gravity", "gval", 0.0);
    int const gdim = reader.GetInteger("Gravity", "gdim", 3);

    KV_double_1d g_array("g_array", ndim);
    Kokkos::parallel_for(ndim, KOKKOS_LAMBDA(int idim)
    {
        g_array(idim) = g * kron(idim, gdim);
    });

    KV_double_1d nodes_x0("nodes_x0", grid.Nx_local_wg[0]+1); // Nodes for x0
    KV_double_1d nodes_y0("nodes_y0", grid.Nx_local_wg[1]+1); // Nodes for y0
    KV_double_1d nodes_z0("nodes_z0", grid.Nx_local_wg[2]+1); // Nodes for z0

    int offsetx = grid.range.Corner_min[0] - grid.Nghost[0];
    Kokkos::parallel_for("InitialisationNodes",
                         Kokkos::RangePolicy<>(0, nodes_x0.extent(0)),
                         KOKKOS_LAMBDA(int i)
    {
        nodes_x0(i) = xmin + (i + offsetx) * array_dx(0) ; // Position of the left interface
    });

    int offsety = grid.range.Corner_min[1] - grid.Nghost[1];
    Kokkos::parallel_for("InitialisationNodes",
                         Kokkos::RangePolicy<>(0, nodes_y0.extent(0)),
                         KOKKOS_LAMBDA(int i)
    {
        nodes_y0(i) = ymin + (i + offsety) * array_dx(1) ; // Position of the left interface
    });
    
    int offsetz = grid.range.Corner_min[2] - grid.Nghost[2];
    Kokkos::parallel_for("InitialisationNodes",
                         Kokkos::RangePolicy<>(0, nodes_z0.extent(0)),
                         KOKKOS_LAMBDA(int i)
    {
        nodes_z0(i) = zmin + (i + offsetz) * array_dx(2) ; // Position of the left interface
    });

    KDV_double_3d rho("rho",  grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2]); // Density
    KDV_double_4d u("u",      grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], ndim); // Speed
    KDV_double_3d P("P",      grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2]); // Pressure
    KV_double_4d rhou("rhou", grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], ndim); // Momentum
    KV_double_3d E("E",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2]); // Energy

    KV_double_5d rho_rec("rho_rec",   grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, ndim);
    KV_double_6d rhou_rec("rhou_rec", grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, ndim, ndim);
    KV_double_5d E_rec("E_rec",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, ndim);
    KV_double_6d u_rec("u_rec",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, ndim, ndim);
    KV_double_5d P_rec("P_rec",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, ndim);

    KV_double_3d rho_new("rhonew",   grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2]);
    KV_double_4d rhou_new("rhounew", grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], ndim);
    KV_double_3d E_new("Enew",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2]);

    initialisation->execute(grid.range.all_ghosts(), rho.d_view, u.d_view, P.d_view, nodes_x0, nodes_y0, g_array);
    conv_prim_to_cons(grid.range.all_ghosts(), rhou, E, rho.d_view, u.d_view, P.d_view, eos);

    double t = 0;
    int iter = 0;
    bool should_exit = false; 

    write_pdi(iter, t, rho, u, P);

    while (!should_exit && t < timeout && iter < max_iter)
    {
        double dt = time_step(grid.range.all_ghosts(), cfl, rho.d_view, u.d_view, P.d_view, array_dx, eos);
        bool const make_output = should_output(iter, output_frequency, max_iter, t, dt, timeout);
        if ((t + dt) > timeout)
        {
            dt = timeout - t;
            should_exit = true;
        }

        face_reconstruction->execute(grid.range.with_ghosts(1), rho.d_view, rho_rec, array_dx);
        face_reconstruction->execute(grid.range.with_ghosts(1), P.d_view, P_rec, array_dx);
        for(int idim = 0; idim < ndim ; ++idim)
        {
            auto u_less_dim = Kokkos::subview(u.d_view, ALL, ALL, ALL, idim);
            auto u_rec_less_dim = Kokkos::subview(u_rec, ALL, ALL, ALL, ALL, ALL, idim);
            face_reconstruction->execute(grid.range.with_ghosts(1), u_less_dim, u_rec_less_dim, array_dx);
        }

        for(int idim = 0; idim < ndim ; ++idim)
        {
            for (int iside = 0; iside < 2; ++iside)
            {
                auto rhou_rec_less_dim = Kokkos::subview(rhou_rec, ALL, ALL, ALL, iside, idim, ALL);
                auto E_rec_less_dim    = Kokkos::subview(E_rec,    ALL, ALL, ALL, iside, idim);
                auto rho_rec_less_dim  = Kokkos::subview(rho_rec,  ALL, ALL, ALL, iside, idim);
                auto u_rec_less_dim    = Kokkos::subview(u_rec,    ALL, ALL, ALL, iside, idim, ALL);
                auto P_rec_less_dim    = Kokkos::subview(P_rec,    ALL, ALL, ALL, iside, idim);
                
                conv_prim_to_cons(grid.range.with_ghosts(1), rhou_rec_less_dim, E_rec_less_dim, rho_rec_less_dim, 
                                        u_rec_less_dim, P_rec_less_dim, eos);
            }
        }

        extrapolation_construction->execute(grid.range.with_ghosts(1), rhou_rec, E_rec, rho_rec, u_rec, P_rec,
                                            eos, array_dx, dt);

        godunov_scheme->execute(grid.range.no_ghosts(), rho.d_view, rhou, E, rho_rec, rhou_rec, E_rec, 
                                rho_new, rhou_new, E_new, array_dx, dt);

        gravity_add->execute(grid.range.no_ghosts(), rho.d_view, rhou, rhou_new, E_new, g_array, dt);

        boundary_construction->execute(rho_new, rhou_new, E_new, grid);

        conv_cons_to_prim(grid.range.all_ghosts(), u.d_view, P.d_view, rho_new, rhou_new, E_new, eos);
        Kokkos::deep_copy(rho.d_view, rho_new);
        Kokkos::deep_copy(rhou, rhou_new);
        Kokkos::deep_copy(E, E_new);
        
        rho.modify_device();
        u.modify_device();
        P.modify_device();

        t += dt;
        iter++;

        if(make_output)
        {
            write_pdi(iter, t, rho, u, P);
        }
    }

    if (grid.mpi_rank == 0)
    {
        std::printf("Final time = %f and number of iterations = %d  \n", t, iter);
        std::printf("--- End ---\n");
    }

    PDI_finalize();
    PC_tree_destroy(&conf);

    return 0;
}
