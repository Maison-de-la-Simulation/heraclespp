/**
 * @file exchange.cpp
 * Holo exchange implementation
 */
#include <array>
#include <mpi.h>
#include "boundary.hpp"

namespace novapp
{


void IBoundaryCondition::ghostFill(KV_double_3d rho,
                                   KV_double_4d rhou,
                                   KV_double_3d E,
                                   KV_double_1d g,
                                   Grid const & grid,
                                   thermodynamics::PerfectGas const& eos)
{
    int ng = grid.Nghost[bc_idim];

    KDV_double_4d buf = grid.mpi_buffer[bc_idim];

    Kokkos::Array<Kokkos::pair<int, int>, 3> KRange = {Kokkos::make_pair(0,rho.extent_int(0)), 
                                                      Kokkos::make_pair(0,rho.extent_int(1)),
                                                      Kokkos::make_pair(0,rho.extent_int(2))};

    KRange[bc_idim].first  = bc_iface==0? ng : rho.extent_int(bc_idim)-2*ng;
    KRange[bc_idim].second = KRange[bc_idim].first + ng;

    Kokkos::deep_copy(Kokkos::subview(buf.d_view, ALL, ALL, ALL, 0),
                      Kokkos::subview(rho, KRange[0], KRange[1], KRange[2]));
    Kokkos::deep_copy(Kokkos::subview(buf.d_view, ALL, ALL, ALL, 1),
                      Kokkos::subview(E, KRange[0], KRange[1], KRange[2]));

    for(int n=0; n<ndim; n++)
    {
        Kokkos::deep_copy(Kokkos::subview(buf.d_view, ALL, ALL, ALL, 2+n),
                          Kokkos::subview(rhou, KRange[0], KRange[1], KRange[2], n));
    }

    buf.modify_device();
    buf.sync_host();
    
    int src, dst;
    MPI_Cart_shift(grid.comm_cart, bc_idim, bc_iface==0?-1:1, &src, &dst);
    
    MPI_Sendrecv_replace(buf.h_view.data(), buf.h_view.size(), MPI_DOUBLE, dst, bc_idim, src, bc_idim, grid.comm_cart, MPI_STATUS_IGNORE);
    
    buf.modify_host();
    buf.sync_device();

    KRange[bc_idim].first  = bc_iface==0? rho.extent_int(bc_idim)-ng : 0;
    KRange[bc_idim].second = KRange[bc_idim].first + ng;
    

    Kokkos::deep_copy(Kokkos::subview(rho, KRange[0], KRange[1], KRange[2]),
                      Kokkos::subview(buf.d_view, ALL, ALL, ALL, 0));

    Kokkos::deep_copy(Kokkos::subview(E, KRange[0], KRange[1], KRange[2]),
                      Kokkos::subview(buf.d_view, ALL, ALL, ALL, 1));

    for(int n=0; n<ndim; n++)
    {
        Kokkos::deep_copy(Kokkos::subview(rhou, KRange[0], KRange[1], KRange[2], n),
                          Kokkos::subview(buf.d_view, ALL, ALL, ALL, 2+n));
    }
}

void BC_init(std::array<std::unique_ptr<IBoundaryCondition>, ndim*2> & BC_array,
             std::array<std::string, ndim*2> & BC_choices,
             Grid const & grid)
{
    for(int idim=0; idim<ndim; idim++)
    {
        for(int iface=0; iface<2; iface++)
        {
            if(!(grid.is_border[idim][iface])) {BC_choices[idim*2+iface] = "Periodic";}
            BC_array[idim*2+iface] = factory_boundary_construction(BC_choices[idim*2+iface], idim, iface);
        }
    }
}


void BC_update(std::array<std::unique_ptr<IBoundaryCondition>, ndim*2> & BC_array, 
               KV_double_3d rho, 
               KV_double_4d rhou, 
               KV_double_3d E,
               KV_double_1d g,
               Grid const & grid,
               thermodynamics::PerfectGas const& eos)
{
    for (int idim=0; idim<ndim; idim++)
    {
        BC_array[idim*2]->ghostFill(rho, rhou, E, g, grid, eos);
        BC_array[idim*2+1]->execute(rho, rhou, E, g, grid, eos);

        BC_array[idim*2+1]->ghostFill(rho, rhou, E, g, grid, eos);
        BC_array[idim*2]->execute(rho, rhou, E, g, grid, eos);
    }
}

} // namespace novapp
