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
#include "float_conversion.hpp"
#include "io.hpp"
#include "face_reconstruction.hpp"
#include "coordinate_system.hpp"
#include "cfl_cond.hpp"
#include "grid.hpp"
#include "set_boundary.hpp"
#include "extrapolation_construction.hpp"
#include "PerfectGas.hpp"
#include "godunov_scheme.hpp"
#include "mpi_scope_guard.hpp"
#include "buffer.hpp"

#include <pdi.h>

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cout << "usage: " << argv[0] << " <path to the ini file> <path to the yaml file>\n";
        return EXIT_FAILURE;
    }

    Kokkos::ScopeGuard guard;
    MpiScopeGuard mpi_guard;
    
    INIReader reader(argv[1]);

    PC_tree_t conf = PC_parse_path(argv[2]);
    PDI_init(PC_get(conf, ".pdi"));

    Grid grid(reader);

    // // a small test
    Buffer send_buffer(&grid, 3);
    Buffer recv_buffer(&grid, 3);

    Kokkos::View<double***> rho3d("rho3D", grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2]);
    Kokkos::deep_copy(rho3d, 1.0);
    copyToBuffer(rho3d, &send_buffer, 0);

    exchangeBuffer(&send_buffer, &recv_buffer, &grid);
    Kokkos::deep_copy(rho3d, 2.0);
    copyFromBuffer(rho3d, &recv_buffer, 0);
    // // end of a small test

    double const timeout = reader.GetReal("Run", "timeout", 0.2);
    int const max_iter = reader.GetInteger("Output", "max_iter", 10000);
    int const output_frequency = reader.GetInteger("Output", "frequency", 10);

    thermodynamics::PerfectGas eos(reader.GetReal("PerfectGas", "gamma", 1.4), 0.0);

    double const dx = 1. / grid.Nx_glob_ng[0];
    double const cfl = 0.4;

    init_write(max_iter, output_frequency, grid.Nghost);

    std::string const initialisation_problem = reader.Get("Problem", "type", "ShockTube");
    std::unique_ptr<IInitialisationProblem> initialisation
            = factory_initialisation(initialisation_problem);

    std::string const reconstruction_type = reader.Get("hydro", "reconstruction", "Minmod");
    std::unique_ptr<IFaceReconstruction> face_reconstruction
            = factory_face_reconstruction(reconstruction_type, dx);

    std::unique_ptr<IExtrapolationValues> extrapolation_construction
            = std::make_unique<ExtrapolationCalculation>();

    std::string const boundary_condition_type = reader.Get("hydro", "boundary", "NullGradient");
    std::unique_ptr<IBoundaryCondition> boundary_construction
            = factory_boundary_construction(boundary_condition_type);
    
    std::string const riemann_solver = reader.Get("hydro", "riemann_solver", "HLL");
    std::unique_ptr<IGodunovScheme> godunov_scheme
            = factory_godunov_scheme(riemann_solver, eos, dx);

    Kokkos::View<double*> nodes_x0("nodes_x0", grid.Nx_local_wg[0]); // Nodes for x0

    Kokkos::parallel_for(
        "initialisation_x0",
        grid.Nx_local_wg[0],
        KOKKOS_LAMBDA(int i)
        {
            nodes_x0(i) = i * dx; // Position of the left interface
        });

    Kokkos::View<double***> rho("rho", grid.Nx_local_wg[0], 1, 1); // Density
    Kokkos::View<double***> rhou("rhou", grid.Nx_local_wg[0], 1, 1); // Momentum
    Kokkos::View<double***> E("E", grid.Nx_local_wg[0], 1, 1); // Energy
    Kokkos::View<double***> u("u", grid.Nx_local_wg[0], 1, 1); // Speed
    Kokkos::View<double***> P("P", grid.Nx_local_wg[0], 1, 1); // Pressure
    
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

    Kokkos::View<double***> rhoL("rhoL", grid.Nx_local_wg[0], 1, 1);
    Kokkos::View<double***> uL("uL", grid.Nx_local_wg[0], 1, 1);
    Kokkos::View<double***> PL("PL", grid.Nx_local_wg[0], 1, 1);
    Kokkos::View<double***> rhoR("rhoR", grid.Nx_local_wg[0], 1, 1);
    Kokkos::View<double***> uR("uR", grid.Nx_local_wg[0], 1, 1);
    Kokkos::View<double***> PR("PR", grid.Nx_local_wg[0], 1, 1);
    Kokkos::View<double***> rhouL("rhouL", grid.Nx_local_wg[0], 1, 1);
    Kokkos::View<double***> EL("EL", grid.Nx_local_wg[0], 1, 1);
    Kokkos::View<double***> rhouR("rhouR", grid.Nx_local_wg[0], 1, 1);
    Kokkos::View<double***> ER("ER", grid.Nx_local_wg[0], 1, 1);

    Kokkos::View<double***> rho_new("rhonew", grid.Nx_local_wg[0], 1, 1);
    Kokkos::View<double***> rhou_new("rhounew", grid.Nx_local_wg[0], 1, 1);
    Kokkos::View<double***> E_new("Enew", grid.Nx_local_wg[0], 1, 1);

    initialisation->execute(rho, u, P, nodes_x0);
    
    ConvPrimConsArray(rhou, E, rho, u, P, eos); // Initialisation conservative variables (rho, rhou, E)
    
    Kokkos::deep_copy(rho_host, rho);
    Kokkos::deep_copy(u_host, u);
    Kokkos::deep_copy(P_host, P);
    Kokkos::deep_copy(rhou_host, rhou);
    Kokkos::deep_copy(E_host, E);
    
    double t = 0;
    int iter = 0;
    bool should_exit = false;

    write(iter, grid.Nx_glob_ng.data(), t, rho.data(), u.data(), P.data());

    while (!should_exit && t < timeout && iter < max_iter)
    {
        double dt = time_step(cfl, rho, u, P, dx, dx, dx, eos);

        bool const make_output = should_output(iter, output_frequency, max_iter, t, dt, timeout);
        
        if ((t + dt) > timeout)
        {
            dt = timeout - t;
            should_exit = true;
        }

        face_reconstruction->execute(rho, rhoL, rhoR); // Calcul des pentes
        face_reconstruction->execute(u, uL, uR);
        face_reconstruction->execute(P, PL, PR);

        ConvPrimConsArray(rhouL, EL, rhoL, uL, PL, eos); // Conversion en variables conservatives
        ConvPrimConsArray(rhouR, ER, rhoR, uR, PR, eos);

        extrapolation_construction->execute(rhoL, uL, PL, rhoR, uR, PR, rhouL, EL, rhouR, ER, eos, dt, dx);
    
        godunov_scheme->execute(
                rho,
                rhou,
                E,
                rhoL,
                rhouL,
                EL,
                rhoR,
                rhouR,
                ER,
                rho_new,
                rhou_new,
                E_new,
                dt);

        boundary_construction->execute(rho_new, rhou_new, E_new, grid.Nghost);

        ConvConsPrimArray(u, P, rho_new, rhou_new, E_new, eos); //Conversion des variables conservatives en primitives
        Kokkos::deep_copy(rho, rho_new);
        Kokkos::deep_copy(rhou, rhou_new);
        Kokkos::deep_copy(E, E_new);

        // {
        // // start border exchange
        // Kokkos::View<double *> s_buf_left("s_buf_l", grid.Nghost);
        // Kokkos::View<double *> s_buf_right("s_buf_r", grid.Nghost);
        // Kokkos::View<double *> r_buf_left("r_buf_l", grid.Nghost);
        // Kokkos::View<double *> r_buf_right("r_buf_r", grid.Nghost);

        // Kokkos::parallel_for(grid.Nghost, KOKKOS_LAMBDA (int i) 
        // { 
        //     s_buf_left(i) = rho(i+grid.Nghost,1,1);
        //     s_buf_right(i) = rho(i+grid.Nx_local_wg[0]-2*grid.Nghost, 1, 1);
        // });

        // MPI_Status mpi_status;    
        // int left_neighbor, right_neighbor;
        
        // MPI_Cart_shift(grid.comm_cart, 0, -1, &right_neighbor, &left_neighbor);
        // //send to left, recv from right
        // MPI_Sendrecv(s_buf_left.data(), 2, MPI_DOUBLE,
        //              left_neighbor, 99,
        //              r_buf_right.data(), 2, MPI_DOUBLE,
        //              right_neighbor, 99,
        //              MPI_COMM_WORLD, &mpi_status);

        // //send to right, recv from left
        // MPI_Sendrecv(s_buf_right.data(), 2, MPI_DOUBLE,
        //              right_neighbor, 99,
        //              r_buf_left.data(), 2, MPI_DOUBLE,
        //              left_neighbor, 99,
        //              MPI_COMM_WORLD, &mpi_status);

        // Kokkos::parallel_for(grid.Nghost, KOKKOS_LAMBDA (int i) 
        // { 
        //     rho(i,1,1) = r_buf_left(i);
        //     rho(i+grid.Nx_local_wg[0]-grid.Nghost, 1, 1) = r_buf_right(i);
        // });
        // // end border exchange
        // }

        t = t + dt;
        iter++;

        if(make_output)
        {
            write(iter, grid.Nx_glob_ng.data(), t, rho.data(), u.data(), P.data());
        }
    }

    std::printf("Final time = %f and number of iterations = %d  \n", t, iter);

    Kokkos::parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {0, 0, 0},
            {grid.Nx_local_wg[0], 1, 1}),
            KOKKOS_LAMBDA(int i, int j, int k)
        {
            std::printf("%f %f %f \n", rho(i, j, k), u(i, j, k), P(i, j, k));
        });
    
    PDI_finalize();
    PC_tree_destroy(&conf);

    std::printf("%s\n", "---Fin du programme---");
    return 0;
}
