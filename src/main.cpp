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

    KV_double_1d dx_array("dx_array", 3); //Space step array
    Kokkos::parallel_for(1, KOKKOS_LAMBDA([[maybe_unused]] int i)
    {
        dx_array(0) = Lx / grid.Nx_glob_ng[0];
        dx_array(1) = Ly / grid.Nx_glob_ng[1];
        dx_array(2) = Lz / grid.Nx_glob_ng[2];
    });

    write_pdi_init(max_iter, output_frequency, grid);

    std::array<std::unique_ptr<IBoundaryCondition>, ndim*2> boundary_construction_array;
    std::array<std::string, 3> bc_dir = {"_X", "_Y", "_Z"};
    std::array<std::string, 3> bc_face = {"_left", "_right"};

    std::string bc_choice;
    std::string bc_choice_dir;
    std::array<std::string, ndim*2> bc_choice_face;
    
    bc_choice = reader.Get("Boundary Condition", "BC", "");
    for(int idim=0; idim<ndim; idim++)
    {
        bc_choice_dir = reader.Get("Boundary Condition", "BC"+bc_dir[idim], bc_choice);
        bc_choice_face[idim*2] = reader.Get("Boundary Condition", "BC"+bc_dir[idim]+bc_face[0], bc_choice_dir);
        bc_choice_face[idim*2+1] = reader.Get("Boundary Condition", "BC"+bc_dir[idim]+bc_face[1], bc_choice_dir);
    }

    for(int idim=0; idim<ndim; idim++)
    {
        if(bc_choice_face[idim*2].empty() || bc_choice_face[idim*2+1].empty()) 
        {
            throw std::runtime_error("boundary condition not defined for dim "+bc_dir[idim]);
        }
        for(int iface=0; iface<2; iface++)
        {
            if(!(grid.is_border[idim][iface])) {bc_choice_face[idim*2+iface] = "Periodic";}
            boundary_construction_array[idim*2+iface] = factory_boundary_construction(grid, bc_choice_face[idim*2+iface], idim, iface);
        }
    }
    
    std::string const initialisation_problem = reader.Get("Problem", "type", "ShockTube");
    std::unique_ptr<IInitialisationProblem> initialisation
            = factory_initialisation(initialisation_problem);

    std::string const reconstruction_type = reader.Get("Hydro", "reconstruction", "VanLeer");
    std::unique_ptr<IFaceReconstruction> face_reconstruction
            = factory_face_reconstruction(reconstruction_type);

    std::string const riemann_solver = reader.Get("Hydro", "riemann_solver", "HLL");
    std::unique_ptr<IGodunovScheme> godunov_scheme
            = factory_godunov_scheme(riemann_solver, eos);

    const double gx = reader.GetReal("Gravity", "gx", 0.0);
    const double gy = reader.GetReal("Gravity", "gy", 0.0);
    const double gz = reader.GetReal("Gravity", "gz", 0.0);

    KV_double_1d g_array("g_array", 3);
    Kokkos::parallel_for(1, KOKKOS_LAMBDA([[maybe_unused]] int i)
    {
        g_array(0) = gx;
        g_array(1) = gy;
        g_array(2) = gz;
    });

    KV_double_1d nodes_x0("nodes_x0", grid.Nx_local_wg[0]+1); // Nodes for x0
    KV_double_1d nodes_y0("nodes_y0", grid.Nx_local_wg[1]+1); // Nodes for y0
    KV_double_1d nodes_z0("nodes_z0", grid.Nx_local_wg[2]+1); // Nodes for z0

    int offsetx = grid.range.Corner_min[0] - grid.Nghost[0];
    Kokkos::parallel_for("InitialisationNodes",
                         Kokkos::RangePolicy<>(0, nodes_x0.extent(0)),
                         KOKKOS_LAMBDA(int i)
    {
        nodes_x0(i) = xmin + (i + offsetx) * dx_array(0) ; // Position of the left interface
    });

    int offsety = grid.range.Corner_min[1] - grid.Nghost[1];
    Kokkos::parallel_for("InitialisationNodes",
                         Kokkos::RangePolicy<>(0, nodes_y0.extent(0)),
                         KOKKOS_LAMBDA(int i)
    {
        nodes_y0(i) = ymin + (i + offsety) * dx_array(1) ; // Position of the left interface
    });

    int offsetz = grid.range.Corner_min[2] - grid.Nghost[2];
    Kokkos::parallel_for("InitialisationNodes",
                         Kokkos::RangePolicy<>(0, nodes_z0.extent(0)),
                         KOKKOS_LAMBDA(int i)
    {
        nodes_z0(i) = zmin + (i + offsetz) * dx_array(2) ; // Position of the left interface
    });

    KDV_double_3d rho("rho",   grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2]); // Density
    KDV_double_4d u("u",       grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], ndim); // Speed
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

    if(reader.GetBoolean("Problem", "restart", false))
    {
        std::string const restart_file = reader.Get("Problem", "restart_file", "restart.h5");
        
        read_pdi(restart_file, rho, u, P, t, iter); // read data into host view
        
        rho.sync_device();
        u.sync_device();
        P.sync_device();
        
        if(grid.mpi_rank==0) 
        {
            std::cout<<std::endl<< std::left << std::setw(80) << std::setfill('*') << "*"<<std::endl;
            std::cout<<"read from file "<<restart_file<<std::endl;
            std::cout<<"starting at time "<<t<<" ( ~ "<<100*t/timeout<<"%)"
                     <<", with iteration "<<iter<<std::endl<<std::endl;
        }
    }
    else
    {
        initialisation->execute(grid.range.no_ghosts(), rho.d_view, u.d_view, P.d_view, nodes_x0, nodes_y0, g_array);
    }
    conv_prim_to_cons(grid.range.no_ghosts(), rhou.d_view, E.d_view, rho.d_view, u.d_view, P.d_view, eos);
    
    boundary_construction_array[0]->ghostFill(rho.d_view, rhou.d_view, E.d_view, grid);
    for ( std::unique_ptr<IBoundaryCondition> const& boundary_construction : boundary_construction_array )
    {
        boundary_construction->execute(rho.d_view, rhou.d_view, E.d_view, grid);
    }
    
    conv_cons_to_prim(grid.range.all_ghosts(), u.d_view, P.d_view, rho.d_view, rhou.d_view, E.d_view, eos);

    std::unique_ptr<IHydroReconstruction> reconstruction = std::make_unique<
            MUSCLHancockHydroReconstruction>(std::move(face_reconstruction), P_rec, u_rec);

    if (output_frequency > 0)
    {
        write_pdi(iter, t, rho, u, P, E);
    }

    while (!should_exit && t < timeout && iter < max_iter)
    {
        double dt = time_step(grid.range.all_ghosts(), cfl, rho.d_view, u.d_view, P.d_view, dx_array, eos);
        bool const make_output = should_output(iter, output_frequency, max_iter, t, dt, timeout);
        if ((t + dt) > timeout)
        {
            dt = timeout - t;
            should_exit = true;
        }

        reconstruction->execute(grid.range.with_ghosts(1), rho_rec, rhou_rec, E_rec, rho.d_view, u.d_view, P.d_view, eos, dx_array, g_array, dt);

        godunov_scheme->execute(grid.range.no_ghosts(), rho.d_view, rhou.d_view, E.d_view, rho_rec, rhou_rec, E_rec,
                                rho_new, rhou_new, E_new, dx_array, g_array, dt);

        boundary_construction_array[0]->ghostFill(rho_new, rhou_new, E_new, grid);
        for ( std::unique_ptr<IBoundaryCondition> const& boundary_construction : boundary_construction_array )
        {
            boundary_construction->execute(rho_new, rhou_new, E_new, grid);
        }

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
            write_pdi(iter, t, rho, u, P, E);
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
