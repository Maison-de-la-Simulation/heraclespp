//!
//! @file io.hpp
//! PDI output functions
//!

#pragma once

#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

#include <kokkos_shortcut.hpp>

namespace novapp
{

class Grid;
class Param;

void print_simulation_status(std::ostream& os, int iter, double current, double time_out);

void write_pdi_init(Grid const& grid, Param const& param);

void write_pdi(
    std::string directory,
    std::string prefix,
    int output_id,
    int iter_output_id,
    int time_output_id,
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
    int& output_id,
    int& iter_output_id,
    int& time_output_id,
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
    int output_id,
    std::vector<std::pair<int, double>> const& outputs_record,
    std::string const& directory,
    std::string const& prefix,
    KDV_double_1d& x,
    KDV_double_1d& y,
    KDV_double_1d& z);
}
