/**
 * @file exchange.cpp
 * Holo exchange implementation
 */

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <inih/INIReader.hpp>

#include <grid.hpp>
#include <kokkos_shortcut.hpp>
#include <ndim.hpp>
#include <nova_params.hpp>

#include "boundary.hpp"
#include "boundary_distribute.hpp"

namespace novapp
{

namespace {

void generate_order(std::array<int, ndim * 2>& bc_order, std::string const& bc_priority)
{
    if (bc_priority.empty()) {
        std::iota(bc_order.begin(), bc_order.end(), 0); // bc_order = {0,1,2,3,4,5};
        return;
    }
    std::array<std::string, ndim * 2> tmp_arr;
    std::stringstream ssin(bc_priority);
    for (int i = 0; i < ndim * 2 && ssin.good(); i++) {
        ssin >> tmp_arr[i];
    }

    int counter = 0;
    for (int i = 0; i < ndim * 2; i++) {
        if (tmp_arr[i] == "X_left") {
            bc_order[0] = i;
            counter++;
        } else if (tmp_arr[i] == "X_right") {
            bc_order[1] = i;
            counter++;
        } else if (tmp_arr[i] == "Y_left" && ndim >= 2) {
            bc_order[2] = i;
            counter++;
        } else if (tmp_arr[i] == "Y_right" && ndim >= 2) {
            bc_order[3] = i;
            counter++;
        } else if (tmp_arr[i] == "Z_left" && ndim == 3) {
            bc_order[4] = i;
            counter++;
        } else if (tmp_arr[i] == "Z_right" && ndim == 3) {
            bc_order[5] = i;
            counter++;
        }
    }
    std::reverse(bc_order.begin(), bc_order.end());
    if (counter != ndim * 2) {
        throw std::runtime_error("boundary priority not fully defined !");
    }
}

} // namespace

void DistributedBoundaryCondition::ghost_sync(
    std::vector<KV_double_3d> const& views,
    int bc_idim,
    int bc_iface) const
{
    int ng = m_grid.Nghost[bc_idim];

    KDV_double_4d buf = m_mpi_buffer[bc_idim];

    Kokkos::Array<Kokkos::pair<int, int>, 3> KRange
            = {Kokkos::make_pair(0, views[0].extent_int(0)),
               Kokkos::make_pair(0, views[0].extent_int(1)),
               Kokkos::make_pair(0, views[0].extent_int(2))};

    KRange[bc_idim].first = bc_iface == 0 ? ng : views[0].extent_int(bc_idim) - 2 * ng;
    KRange[bc_idim].second = KRange[bc_idim].first + ng;

    for (std::size_t i = 0; i < views.size(); ++i)
    {
        Kokkos::deep_copy(
                Kokkos::subview(buf.d_view, ALL, ALL, ALL, i),
                Kokkos::subview(views[i], KRange[0], KRange[1], KRange[2]));
    }

    buf.modify_device();

    double* ptr = nullptr;
    if (!m_param.mpi_device_aware)
    {
        buf.sync_host();
        ptr = buf.h_view.data();
    }
    else
    {
        ptr = buf.d_view.data();
    }

    int src, dst;
    MPI_Cart_shift(m_grid.comm_cart, bc_idim, bc_iface == 0 ? -1 : 1, &src, &dst);

    MPI_Sendrecv_replace(
            ptr,
            buf.h_view.size(),
            MPI_DOUBLE,
            dst,
            bc_idim,
            src,
            bc_idim,
            m_grid.comm_cart,
            MPI_STATUS_IGNORE);

    if (!m_param.mpi_device_aware)
    {
        buf.modify_host();
        buf.sync_device();
    }

    KRange[bc_idim].first = bc_iface == 0 ? views[0].extent_int(bc_idim) - ng : 0;
    KRange[bc_idim].second = KRange[bc_idim].first + ng;

    for (std::size_t i = 0; i < views.size(); ++i)
    {
        Kokkos::deep_copy(
                Kokkos::subview(views[i], KRange[0], KRange[1], KRange[2]),
                Kokkos::subview(buf.d_view, ALL, ALL, ALL, i));
    }
}

DistributedBoundaryCondition::DistributedBoundaryCondition(
    Grid const& grid, 
    Param const& param,
    std::array<std::unique_ptr<IBoundaryCondition>, ndim * 2> bcs)
    : m_bcs(std::move(bcs))
    , m_grid(grid)
    , m_param(param)
{
    for(int idim=0; idim<ndim; idim++)
    {
        for(int iface=0; iface<2; iface++)
        {
            if (!m_grid.is_border[idim][iface])
            {
                m_bcs[idim * 2 + iface] = std::make_unique<PeriodicCondition>(idim, iface);
            }
        }
    }

    std::array<int, 3> buf_size = grid.Nx_local_wg;
    for (int idim = 0; idim < ndim; idim++)
    {
        buf_size[idim] = grid.Nghost[idim];
        m_mpi_buffer[idim] = KDV_double_4d("", buf_size[0], buf_size[1], buf_size[2], ndim + 2 + param.nfx);
        buf_size[idim] = grid.Nx_local_wg[idim];
    }

    generate_order(m_bc_order, m_param.bc_priority);
}

void DistributedBoundaryCondition::operator()(KV_double_3d rho,
                                              KV_double_4d rhou,
                                              KV_double_3d E,
                                              KV_double_4d fx) const
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

    for (int idim = 0; idim < ndim; idim++)
    {
        for (int iface = 0; iface < 2; iface++)
        {
            ghost_sync(views, idim, iface);
        }
    }

    for ( int const bc_id : m_bc_order )
    {
        m_bcs[bc_id]->execute(rho, rhou, E, fx);
    }
}

} // namespace novapp
