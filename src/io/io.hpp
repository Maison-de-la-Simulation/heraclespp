//!
//! @file io.hpp
//! PDI output functions
//!

#pragma once

#include <string>
#include <utility>
#include <vector>

#include <kokkos_shortcut.hpp>

namespace novapp
{

class Grid;
class Param;

class ShouldOutput
{
    int m_freq;
    int m_iter_max;
    double m_time_out;

public:
    ShouldOutput(int freq, int iter_max, double time_out);

    [[nodiscard]] bool operator()(int iter, double current, double dt) const;
};

void write_pdi_init(
    std::string directory,
    std::string prefix,
    int max_iter,
    int frequency,
    Grid const& grid,
    Param const& param);

void write_pdi(
    std::string directory,
    std::string prefix,
    int iter,
    double t,
    double gamma,
    KDV_double_3d& rho,
    KDV_double_4d& u,
    KDV_double_3d& P,
    KDV_double_3d& E,
    KDV_double_1d& x,
    KDV_double_1d& y,
    KDV_double_1d& z,
    KDV_double_4d& fx,
    KDV_double_3d& T);

void read_pdi(
    std::string restart_file,
    int& iter,
    double& t,
    KDV_double_3d& rho,
    KDV_double_4d& u,
    KDV_double_3d& P,
    KDV_double_4d& fx,
    KDV_double_1d& x_glob,
    KDV_double_1d& y_glob,
    KDV_double_1d& z_glob);

void write_xml(
    Grid const& grid,
    std::vector<std::pair<int, double>> const& outputs_record,
    std::string const& directory,
    std::string const& prefix,
    KDV_double_1d& x,
    KDV_double_1d& y,
    KDV_double_1d& z);
}
