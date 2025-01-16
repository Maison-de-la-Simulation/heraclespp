#pragma once

#include <array>
#include <string>

#include <inih/INIReader.hpp>

namespace novapp
{

class Param
{
public :
    INIReader reader;
    std::string problem;
    bool restart;
    std::string restart_file;
    std::array<int, 3> Nx_glob_ng;
    double xmin;
    double xmax;
    double ymin;
    double ymax;
    double zmin;
    double zmax;
    std::string grid_type;
    int Ng;
    std::array<int,3> mpi_dims_cart;
    double t_ini;
    double t_end;
    double cfl;
    int max_iter;
    int iter_output_frequency;
    double time_output_frequency;
    std::string directory;
    std::string prefix;
    std::string reconstruction_type;
    std::string riemann_solver;
    double gx;
    double gy;
    double gz;
    double M;
    double gamma;
    double mu;
    std::string bc_choice;
    std::string bc_priority;
    int nfx;
    std::string user_step;
    bool mpi_device_aware;
    std::string pressure_fix;
    double eps_pf;
    int time_job;
    std::string shift_grid;
    double rmax_shift;

    explicit Param(INIReader const& reader);

    Param(Param const& rhs);

    Param(Param&& rhs) noexcept;

    ~Param() noexcept;

    Param& operator=(Param const& rhs);

    Param& operator=(Param&& rhs) noexcept;
};

} // namespace novapp
