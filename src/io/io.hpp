// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

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

namespace hclpp {

class Grid;
class Param;

void print_simulation_status(std::ostream& os, int iter, double current, double time_out, int output_id);

void write_pdi_init(Grid const& grid, Param const& param);

void write_pdi(
        std::string const& directory,
        std::string const& prefix,
        int output_id,
        int iter_output_id,
        int time_output_id,
        int iter,
        double t,
        double gamma,
        Grid const& grid,
        KDV_cdouble_3d const& rho,
        KDV_cdouble_4d const& u,
        KDV_cdouble_3d const& P,
        KDV_cdouble_3d const& E,
        KDV_cdouble_1d const& x0,
        KDV_cdouble_1d const& x1,
        KDV_cdouble_1d const& x2,
        KDV_cdouble_4d const& fx,
        KDV_cdouble_3d const& T);

void read_pdi(
        std::string const& restart_file,
        int& output_id,
        int& iter_output_id,
        int& time_output_id,
        int& iter,
        double& t,
        KDV_double_3d const& rho,
        KDV_double_4d const& u,
        KDV_double_3d const& P,
        KDV_double_4d const& fx,
        KDV_double_1d const& x0_glob,
        KDV_double_1d const& x1_glob,
        KDV_double_1d const& x2_glob);

class XmlWriter
{
    std::string m_directory;

    std::string m_prefix;

    std::vector<std::string> m_var_names;

public:
    XmlWriter(std::string directory, std::string prefix, int nfx);

    void operator()(
            Grid const& grid,
            int output_id,
            std::vector<std::pair<int, double>> const& outputs_record,
            KDV_cdouble_1d const& x0,
            KDV_cdouble_1d const& x1,
            KDV_cdouble_1d const& x2) const;
};

} // namespace hclpp
