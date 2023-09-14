//!
//! @file boundary_factory.hpp
//!

#pragma once

#include <array>
#include <memory>

#include <inih/INIReader.hpp>

#include "mesh/grid.hpp"
#include "kokkos_shortcut.hpp"
#include "ndim.hpp"
#include "nova_params.hpp"

namespace novapp
{

class IBoundaryCondition;

class DistributedBoundaryCondition
{
private:
    std::array<KDV_double_4d, ndim> m_mpi_buffer;
    std::array<std::unique_ptr<IBoundaryCondition>, ndim * 2> m_bcs;
    std::array<int, ndim*2> m_bc_order;
    Grid m_grid;
    Param m_param;

    void ghost_sync(std::vector<KV_double_3d> const& views, int bc_idim, int bc_iface) const;

public:
    DistributedBoundaryCondition(
            Grid const& grid,
            Param const& param,
            std::array<std::unique_ptr<IBoundaryCondition>, ndim * 2> bcs);

    void operator()(KV_double_3d rho,
                    KV_double_4d rhou,
                    KV_double_3d E,
                    KV_double_4d fx) const;
};

} // namespace novapp
