//!
//! @file nova_params.hpp
//! Grid class declaration
//!

#pragma once

namespace novapp
{

class Param
{
public :
    std::string initialisation_problem;
    bool restart;
    std::string restart_file;
    std::array<int, 3> Nx_glob_ng;
    double xmin;
    double xmax;
    double ymin;
    double ymax;
    double zmin;
    double zmax;
    int Ng;
    std::array<int,3> Ncpu_x;
    std::string system;
    double timeout;
    double cfl;
    int max_iter;
    int output_frequency;
    std::string reconstruction_type;
    std::string riemann_solver;    
    std::string gravity_type;
    double gx;
    double gy;
    double gz;
    double gamma;
    double T;
    double mu;
    std::string bc_choice;
    std::string bc_priority;
    double rho0;
    double rho1;
    double u0;
    double u1;
    double P0;
    double P1;
    double E0;
    double E1;
    double A;
    double nfx;

    explicit Param(INIReader const& reader)
    {
        initialisation_problem = reader.Get("Problem", "type", "ShockTube");
        restart = reader.GetBoolean("Problem", "restart", false);
        restart_file = reader.Get("Problem", "restart_file", "restart.h5");

        Nx_glob_ng[0] = reader.GetInteger("Grid", "Nx_glob", 0); // Cell number
        Nx_glob_ng[1] = reader.GetInteger("Grid", "Ny_glob", 0); // Cell number
        Nx_glob_ng[2] = reader.GetInteger("Grid", "Nz_glob", 0); // Cell number

        Ng = reader.GetInteger("Grid", "Nghost", 2);

        xmin = reader.GetReal("Grid", "xmin", 0.0);
        xmax = reader.GetReal("Grid", "xmax", 1.0);
        ymin = reader.GetReal("Grid", "ymin", 0.0);
        ymax = reader.GetReal("Grid", "ymax", 1.0);
        zmin = reader.GetReal("Grid", "zmin", 0.0);
        zmax = reader.GetReal("Grid", "zmax", 1.0);

        Ncpu_x[0] = reader.GetInteger("Grid", "Ncpu_x", 0); // number of procs, default 0=>defined by MPI
        Ncpu_x[1] = reader.GetInteger("Grid", "Ncpu_y", 0); // number of procs
        Ncpu_x[2] = reader.GetInteger("Grid", "Ncpu_z", 0); // number of procs

        system = reader.Get("Coordinate", "system", "Cartesian");

        timeout = reader.GetReal("Run", "timeout", 0.2);
        cfl = reader.GetReal("Run", "cfl", 0.4);

        max_iter = reader.GetInteger("Output", "max_iter", 10000);
        output_frequency = reader.GetInteger("Output", "frequency", 10);

        reconstruction_type = reader.Get("Hydro", "reconstruction", "VanLeer");
        riemann_solver = reader.Get("Hydro", "riemann_solver", "HLL");

        gravity_type = reader.Get("Gravity", "type", "Uniform");
        gx = reader.GetReal("Gravity", "gx", 0.0);
        gy = reader.GetReal("Gravity", "gy", 0.0);
        gz = reader.GetReal("Gravity", "gz", 0.0);

        gamma = reader.GetReal("Perfect Gas", "gamma", 5./3);
        T = reader.GetReal("Perfect Gas", "temperature", 100.);
        mu = reader.GetReal("Perfect Gas", "mu", 1.);

        bc_choice = reader.Get("Boundary Condition", "BC", "");
        bc_priority = reader.Get("Boundary Condition", "priority", "");

        nfx = reader.GetInteger("Passive Scalar", "nfx", 0);

        // à mettre ailleurs
        rho0 = reader.GetReal("Initialisation", "rho0", 1.0);
        rho1 = reader.GetReal("Initialisation", "rho1", 1.0);
        u0 = reader.GetReal("Initialisation", "u0", 1.0);
        u1 = reader.GetReal("Initialisation", "u1", 1.0);
        P0 = reader.GetReal("Initialisation", "P0", 1.0);
        P1 = reader.GetReal("Initialisation", "P1", 1.0);
        E0 = reader.GetReal("Initialisation", "E0", 1.0);
        E1 = reader.GetReal("Initialisation", "E1", 1.0);
        A = reader.GetReal("Initialisation", "A", 1.0); // amplitude
    }
};

} // namespace novapp