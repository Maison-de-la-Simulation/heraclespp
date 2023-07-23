//!
//! @file io.hpp
//! PDI output functions
//!

#pragma once

#include <array>
#include "grid.hpp"
#include "kokkos_shortcut.hpp"

namespace novapp
{

bool should_output(int iter, int freq, int iter_max, double current, double dt, double time_out);

void write_pdi_init(int max_iter, int frequency, Grid const& grid, Param const& param);

void write_pdi(int iter,
               double t,
               double gamma,
               KDV_double_3d rho, 
               KDV_double_4d u,
               KDV_double_3d P,
               KDV_double_3d E,
               KDV_double_1d x,
               KDV_double_1d y,
               KDV_double_1d z,
               KDV_double_4d fx);

void read_pdi(std::string restart_file,
              KDV_double_3d rho,
              KDV_double_4d u,
              KDV_double_3d P,
              KDV_double_4d fx,
              double& t,
              int& iter);

void writeXML(
        Grid const& grid,
        std::vector<std::pair<int, double>> const& outputs_record,
        KVH_double_1d x,
        KVH_double_1d y,
        KVH_double_1d z);
}
