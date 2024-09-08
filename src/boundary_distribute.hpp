//!
//! @file boundary_factory.hpp
//!

#pragma once

#include <array>
#include <memory>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <grid.hpp>
#include <kokkos_shortcut.hpp>
#include <ndim.hpp>
#include <nova_params.hpp>

#include "boundary.hpp"

namespace novapp
{

class DistributedBoundaryCondition
{
    using mpi_buffer_type = Kokkos::DualView<double****, Kokkos::LayoutLeft, Kokkos::SharedHostPinnedSpace>;

private:
    std::array<mpi_buffer_type, ndim> m_mpi_buffer;
    std::array<int, ndim*2> m_bc_order;
    Param m_param;

    void ghost_sync(Grid const& grid, std::vector<KV_double_3d> const& views, int bc_idim, int bc_iface) const;

public:
    DistributedBoundaryCondition(
            Grid const& grid,
            Param const& param);

    template <class Gravity>
    void operator()(std::array<std::unique_ptr<IBoundaryCondition<Gravity>>, ndim * 2> const& bcs,
                    Grid const& grid,
                    Gravity const& gravity,
                    KV_double_3d const& rho,
                    KV_double_4d const& rhou,
                    KV_double_3d const& E,
                    KV_double_4d const& fx) const
    {
        std::vector<KV_double_3d> views;
        views.reserve(2 + rhou.extent_int(3) + fx.extent_int(3));
        views.emplace_back(rho);
        for (int i3 = 0; i3 < rhou.extent_int(3); ++i3)
        {
            views.emplace_back(Kokkos::subview(rhou, ALL, ALL, ALL, i3));
        }
        views.emplace_back(E);
        for (int i3 = 0; i3 < fx.extent_int(3); ++i3)
        {
            views.emplace_back(Kokkos::subview(fx, ALL, ALL, ALL, i3));
        }

        for (int idim = 0; idim < ndim; ++idim)
        {
            for (int iface = 0; iface < 2; ++iface)
            {
                ghost_sync(grid, views, idim, iface);
            }
        }

        for ( int const bc_id : m_bc_order )
        {
            int const idim = bc_id / 2;
            int const iface = bc_id % 2;
            if (grid.is_border[idim][iface])
            {
                bcs[bc_id]->execute(grid, gravity, rho, rhou, E, fx);
            }
        }
    }
};

} // namespace novapp
