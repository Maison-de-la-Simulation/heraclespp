//!
//! @file boundary_factory.hpp
//!

#pragma once

#include <algorithm>
#include <iostream>
#include <numeric>
#include <sstream>

#include <PerfectGas.hpp>

#include "ndim.hpp"
#include "units.hpp"
#include "eos.hpp"
#include "kokkos_shortcut.hpp"
#include "grid.hpp"
#include "nova_params.hpp"
#include "boundary.hpp"
#include "factories.hpp"
#include "setup.hpp"

namespace novapp
{

class DistributedBoundaryCondition
{
private:
    std::array<KDV_double_4d, ndim> m_mpi_buffer;
    std::array<std::unique_ptr<IBoundaryCondition>, ndim * 2> m_bcs;
    std::array<int, ndim*2> m_bc_order;
    EOS m_eos;
    Grid m_grid;
    Param m_param;
    ParamSetup m_param_setup;

    void ghostFill(
            KV_double_3d rho,
            KV_double_4d rhou,
            KV_double_3d E,
            KV_double_4d fx,
            int bc_idim,
            int bc_iface) const;

    void generate_order();

public:
    DistributedBoundaryCondition(
            INIReader const& reader,
            EOS const& eos,
            Grid const& grid,
            Param const& param,
            ParamSetup const& param_setup);

    void execute(KV_double_3d rho,
                 KV_double_4d rhou,
                 KV_double_3d E,
                 KV_double_4d fx,
                 KV_double_1d g) const;
};

} // namespace novapp
