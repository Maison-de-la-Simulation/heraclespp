/**
 * @file exchange.cpp
 * Holo exchange implementation
 */

#include <array>
#include <mpi.h>
#include <vector>

#include "boundary_distribute.hpp"

namespace novapp
{

void DistributedBoundaryCondition::ghostFill(
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
    buf.sync_host();

    int src, dst;
    MPI_Cart_shift(m_grid.comm_cart, bc_idim, bc_iface == 0 ? -1 : 1, &src, &dst);

    MPI_Sendrecv_replace(
            buf.h_view.data(),
            buf.h_view.size(),
            MPI_DOUBLE,
            dst,
            bc_idim,
            src,
            bc_idim,
            m_grid.comm_cart,
            MPI_STATUS_IGNORE);

    buf.modify_host();
    buf.sync_device();

    KRange[bc_idim].first = bc_iface == 0 ? views[0].extent_int(bc_idim) - ng : 0;
    KRange[bc_idim].second = KRange[bc_idim].first + ng;

    for (std::size_t i = 0; i < views.size(); ++i)
    {
        Kokkos::deep_copy(
                Kokkos::subview(views[i], KRange[0], KRange[1], KRange[2]),
                Kokkos::subview(buf.d_view, ALL, ALL, ALL, i));
    }
}

void DistributedBoundaryCondition::generate_order()
{
    if(m_param.bc_priority.empty())
    {
        std::iota(m_bc_order.begin(), m_bc_order.end(), 0); // m_bc_order = {0,1,2,3,4,5};
        return;
    }
    std::array<std::string, ndim*2> tmp_arr;
    std::stringstream ssin(m_param.bc_priority);
    for(int i=0; i<ndim*2 && ssin.good(); i++)
    {
        ssin >> tmp_arr[i];
    }

    int counter=0;
    for(int i=0; i<ndim*2; i++)
    {
        if(tmp_arr[i] == "X_left") {m_bc_order[0] = i; counter++;}
        else if(tmp_arr[i] == "X_right") {m_bc_order[1] = i; counter++;}
        else if(tmp_arr[i] == "Y_left" && ndim>=2) {m_bc_order[2] = i; counter++;}
        else if(tmp_arr[i] == "Y_right" && ndim>=2) {m_bc_order[3] = i; counter++;}
        else if(tmp_arr[i] == "Z_left" && ndim==3) {m_bc_order[4] = i; counter++;}
        else if(tmp_arr[i] == "Z_right" && ndim==3) {m_bc_order[5] = i; counter++;}
    }
    std::reverse(m_bc_order.begin(), m_bc_order.end());
    if(counter!=ndim*2)
    {
        throw std::runtime_error("boundary priority not fully defined !");
    }
}

DistributedBoundaryCondition::DistributedBoundaryCondition(
    INIReader const& reader,
    EOS const& eos,
    Grid const& grid, 
    Param const& param,
    ParamSetup const& param_setup)
    :  m_eos(eos)
    , m_grid(grid)
    , m_param(param)
    , m_param_setup(param_setup)
{
    std::string bc_choice_dir;
    std::array<std::string, ndim*2> bc_choice_faces;

    for(int idim = 0; idim < ndim; idim++)
    {
        bc_choice_dir = reader.Get("Boundary Condition", "BC"+bc_dir[idim], m_param.bc_choice);
        for (int iface = 0; iface < 2; iface++)
        {
            bc_choice_faces[idim*2+iface] = reader.Get("Boundary Condition",
                                                       "BC"+bc_dir[idim]+bc_face[iface],
                                                       bc_choice_dir);
            if(bc_choice_faces[idim*2+iface].empty() )
            {
                throw std::runtime_error("boundary condition not fully defined for dimension "
                                         +bc_dir[idim]);
            }
        }
    }

    for(int idim=0; idim<ndim; idim++)
    {
        for(int iface=0; iface<2; iface++)
        {
            if (!m_grid.is_border[idim][iface])
            {
                m_bcs[idim * 2 + iface] = factory_boundary_construction(
                    "Periodic", idim, iface, m_eos, m_grid, m_param_setup);
            }
            else
            {
                m_bcs[idim*2+iface] = factory_boundary_construction(
                    bc_choice_faces[idim*2+iface], idim, iface, m_eos, m_grid, m_param_setup);
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

    generate_order();
}

void DistributedBoundaryCondition::execute(KV_double_3d rho,
                                           KV_double_4d rhou,
                                           KV_double_3d E,
                                           KV_double_4d fx,
                                           KV_double_1d g) const
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
            ghostFill(views, idim, iface);
        }
    }

    for ( int const bc_id : m_bc_order )
    {
        m_bcs[bc_id]->execute(rho, rhou, E, fx, g);
    }
}

} // namespace novapp
