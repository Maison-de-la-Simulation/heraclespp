//!
//! @file boundary_factory.hpp
//!

#pragma once

#include <array>
#include <memory>

#include <inih/INIReader.hpp>

#include <grid.hpp>
#include <kokkos_shortcut.hpp>
#include <ndim.hpp>
#include <nova_params.hpp>

namespace novapp
{

class IBoundaryCondition;

class DistributedBoundaryCondition
{
    using mpi_buffer_type = Kokkos::DualView<double****, Kokkos::LayoutLeft, Kokkos::SharedHostPinnedSpace>;

private:
    std::array<mpi_buffer_type, ndim> m_mpi_buffer;
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

    void operator()(KV_double_3d const& rho,
                    KV_double_4d const& rhou,
                    KV_double_3d const& E,
                    KV_double_4d const& fx) const;
};

} // namespace novapp
