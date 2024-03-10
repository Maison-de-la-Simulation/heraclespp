#include <array>
#include <string>

#include <inih/INIReader.hpp>

#include "nova_params.hpp"

namespace novapp
{

Param::Param(INIReader const& reader)
    : reader(reader)
{
    problem = reader.Get("Problem", "type", "ShockTube");
    restart = reader.GetBoolean("Problem", "restart", false);
    restart_file = reader.Get("Problem", "restart_file", "restart.h5");

    Nx_glob_ng[0] = reader.GetInteger("Grid", "Nx_glob", 0); // Cell number
    Nx_glob_ng[1] = reader.GetInteger("Grid", "Ny_glob", 0); // Cell number
    Nx_glob_ng[2] = reader.GetInteger("Grid", "Nz_glob", 0); // Cell number
    grid_type = reader.Get("Grid", "type", "Regular");
    Ng = reader.GetInteger("Grid", "Nghost", 2);
    xmin = reader.GetReal("Grid", "xmin", 0.0);
    xmax = reader.GetReal("Grid", "xmax", 1.0);
    ymin = reader.GetReal("Grid", "ymin", 0.0);
    ymax = reader.GetReal("Grid", "ymax", 1.0);
    zmin = reader.GetReal("Grid", "zmin", 0.0);
    zmax = reader.GetReal("Grid", "zmax", 1.0);

    mpi_device_aware = reader.GetBoolean("Parallelization", "mpi_device_aware", false);
    Ncpu_x[0] = reader.GetInteger("Parallelization", "Ncpu_x", 0); // number of procs, default 0=>defined by MPI
    Ncpu_x[1] = reader.GetInteger("Parallelization", "Ncpu_y", 0); // number of procs
    Ncpu_x[2] = reader.GetInteger("Parallelization", "Ncpu_z", 0); // number of procs

    t_ini = reader.GetReal("Run", "t_ini", 0.);
    t_end = reader.GetReal("Run", "t_end", 0.2);
    cfl = reader.GetReal("Run", "cfl", 0.4);

    max_iter = reader.GetInteger("Output", "max_iter", 10000);
    output_frequency = reader.GetInteger("Output", "frequency", 10);
    directory = reader.Get("Output", "directory", "build");
    prefix = reader.Get("Output", "prefix", "result");

    reconstruction_type = reader.Get("Hydro", "reconstruction", "VanLeer");
    riemann_solver = reader.Get("Hydro", "riemann_solver", "HLL");

    gx = reader.GetReal("Gravity", "gx", 0.0);
    gy = reader.GetReal("Gravity", "gy", 0.0);
    gz = reader.GetReal("Gravity", "gz", 0.0);
    M = reader.GetReal("Gravity", "M", 1.0);

    gamma = reader.GetReal("Perfect Gas", "gamma", 5./3);
    mu = reader.GetReal("Perfect Gas", "mu", 1.);

    bc_choice = reader.Get("Boundary Condition", "BC", "");
    bc_priority = reader.Get("Boundary Condition", "priority", "");

    nfx = reader.GetInteger("Passive Scalar", "nfx", 0);

    user_step = reader.Get("User step", "user_step", "Off");

    pressure_fix = reader.Get("Pressure fix", "pressure_fix", "Off");
    eps_pf = reader.GetReal("Pressure fix", "eps_pf", 0.000001);
}

Param::Param(Param const& rhs) = default;

Param::Param(Param&& rhs) noexcept = default;

Param::~Param() noexcept = default;

Param& Param::operator=(Param const& rhs) = default;

Param& Param::operator=(Param&& rhs) noexcept = default;

} // namespace novapp
