// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <array>
#include <string>

#include <inih/INIReader.hpp>

#include "param.hpp"

namespace hclpp {

Param::Param(INIReader const& reader) : reader(reader)
{
    // NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    problem = reader.Get("Problem", "type", "ShockTube");
    restart = reader.GetBoolean("Problem", "restart", false);
    restart_file = reader.Get("Problem", "restart_file", "None");
    rmax_shift = reader.GetReal("Problem", "rmax_shift", 1E20);

    Nx_glob_ng[0] = reader.GetInteger("Grid", "Nx0_glob", 0); // Cell number
    Nx_glob_ng[1] = reader.GetInteger("Grid", "Nx1_glob", 0); // Cell number
    Nx_glob_ng[2] = reader.GetInteger("Grid", "Nx2_glob", 0); // Cell number
    grid_type = reader.Get("Grid", "type", "Regular");
    Ng = reader.GetInteger("Grid", "Nghost", 2);
    x0min = reader.GetReal("Grid", "x0min", 0.0);
    x0max = reader.GetReal("Grid", "x0max", 1.0);
    x1min = reader.GetReal("Grid", "x1min", 0.0);
    x1max = reader.GetReal("Grid", "x1max", 1.0);
    x2min = reader.GetReal("Grid", "x2min", 0.0);
    x2max = reader.GetReal("Grid", "x2max", 1.0);
    shift_grid = reader.Get("Grid", "shift_grid", "Off");

    mpi_device_aware = reader.GetBoolean("Parallelization", "mpi_device_aware", false);
    mpi_dims_cart[0] = reader.GetInteger("Parallelization", "mpi_dims_cart_x0", 0); // number of procs, default 0=>defined by MPI
    mpi_dims_cart[1] = reader.GetInteger("Parallelization", "mpi_dims_cart_x1", 0); // number of procs
    mpi_dims_cart[2] = reader.GetInteger("Parallelization", "mpi_dims_cart_x2", 0); // number of procs

    t_ini = reader.GetReal("Run", "t_ini", 0.);
    t_end = reader.GetReal("Run", "t_end", 0.2);
    cfl = reader.GetReal("Run", "cfl", 0.4);

    max_iter = reader.GetInteger("Output", "max_iter", 10000);
    iter_output_frequency = reader.GetInteger("Output", "iter_frequency", 0);
    time_output_frequency = reader.GetReal("Output", "time_frequency", 0.0);
    time_first_output = reader.GetReal("Output", "time_first_output", t_ini + time_output_frequency);
    directory = reader.Get("Output", "directory", ".");
    prefix = reader.Get("Output", "prefix", "result");
    time_job = reader.GetInteger("Output", "time_job", 20);

    reconstruction_type = reader.Get("Hydro", "reconstruction", "VanLeer");
    riemann_solver = reader.Get("Hydro", "riemann_solver", "HLL");

    gx0 = reader.GetReal("Gravity", "gx0", 0.0);
    gx1 = reader.GetReal("Gravity", "gx1", 0.0);
    gx2 = reader.GetReal("Gravity", "gx2", 0.0);
    M = reader.GetReal("Gravity", "M", 1.0);

    gamma = reader.GetReal("Perfect Gas", "gamma", 5. / 3);
    mu = reader.GetReal("Perfect Gas", "mu", 1.);

    bc_choice = reader.Get("Boundary Condition", "BC", "");
    bc_priority = reader.Get("Boundary Condition", "priority", "");

    nfx = reader.GetInteger("Passive Scalar", "nfx", 0);

    user_step = reader.Get("User step", "user_step", "Off");

    pressure_fix = reader.Get("Pressure fix", "pressure_fix", "Off");
    eps_pf = reader.GetReal("Pressure fix", "eps_pf", 0.000001);
    // NOLINTEND(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
}

Param::Param(Param const& rhs) = default;

Param::Param(Param&& rhs) noexcept = default;

Param::~Param() noexcept = default;

Param& Param::operator=(Param const& rhs) = default;

Param& Param::operator=(Param&& rhs) noexcept = default;

} // namespace hclpp
