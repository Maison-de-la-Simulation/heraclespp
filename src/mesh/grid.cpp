// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

/**
 * @file grid.cpp
 * Grid class implementation
 */

#include <mpi.h>

#include <array>
#include <cassert>
#include <memory>
#include <string>

#include <Kokkos_Core.hpp>
#include <kokkos_shortcut.hpp>
#include <ndim.hpp>
#include <print_info.hpp>

#include "geometry.hpp"
#include "geometry_factory.hpp"
#include "grid.hpp"
#include "range.hpp"

namespace novapp
{

namespace
{

void compute_cell_size(KV_cdouble_1d const& x, KV_double_1d const& dx)
{
    assert(x.extent_int(0) == dx.extent_int(0) + 1);
    Kokkos::parallel_for(
        "fill_cell_size_array",
        Kokkos::RangePolicy<int>(0, dx.extent_int(0)),
        KOKKOS_LAMBDA(int i)
        {
            dx(i) = x(i+1) - x(i);
        });
}

void compute_cell_center(KV_cdouble_1d const& x, KV_double_1d const& x_center)
{
    assert(x.extent_int(0) == x_center.extent_int(0) + 1);
    Kokkos::parallel_for(
        "fill_cell_center_array",
        Kokkos::RangePolicy<int>(0, x_center.extent_int(0)),
        KOKKOS_LAMBDA(int i)
        {
            x_center(i) = (x(i) + x(i+1)) / 2;
        });
}

} // namespace

Grid::Grid(std::array<int, 3> const& nx_glob_ng, std::array<int, 3> const& mpi_dims_cart, int const Ng)
    : Ng(Ng)
    , Nghost {0, 0, 0}
    , Nx_glob_ng(nx_glob_ng)
    , Nx_local_ng(nx_glob_ng)
    , mpi_rank_cart {0, 0, 0}
    , mpi_dims_cart(mpi_dims_cart)
{
    for (int idim = 0; idim < ndim; ++idim)
    {
        Nghost[idim] = Ng;
    }

    mpi_decomposition();

    x = KV_double_1d("x", Nx_local_wg[0]+1);
    y = KV_double_1d("y", Nx_local_wg[1]+1);
    z = KV_double_1d("z", Nx_local_wg[2]+1);

    x_center = KV_double_1d("x_center", Nx_local_wg[0]);
    y_center = KV_double_1d("y_center", Nx_local_wg[1]);
    z_center = KV_double_1d("z_center", Nx_local_wg[2]);

    dx = KV_double_1d("dx", Nx_local_wg[0]);
    dy = KV_double_1d("dy", Nx_local_wg[1]);
    dz = KV_double_1d("dz", Nx_local_wg[2]);

    ds = KV_double_4d("ds", Nx_local_wg[0],
                            Nx_local_wg[1],
                            Nx_local_wg[2],
                            3);
    dv = KV_double_3d("dv", Nx_local_wg[0],
                            Nx_local_wg[1],
                            Nx_local_wg[2]);
}

Grid::~Grid() noexcept
{
    if (comm_cart != MPI_COMM_NULL) {
        MPI_Comm_free(&comm_cart);
    }
}

/* ****************************************************************
This routine distribute the cpu over the various direction in an optimum way
mpi_dims_cart  : Number of cpu along each direction, output
          mpi_size = mpi_dims_cart[0] * mpi_dims_cart[1] * mpi_dims_cart[2]
*/
void Grid::mpi_decomposition()
{
    int world_size = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Dims_create(world_size, ndim, mpi_dims_cart.data());
    for(int n=ndim; n<3; ++n)
    {
        mpi_dims_cart[n] = 1;
    }
    std::array<int, 3> periodic = {1,1,1};

    MPI_Cart_create(MPI_COMM_WORLD, 3, mpi_dims_cart.data(), periodic.data(), 1, &comm_cart);
    MPI_Comm_size(comm_cart, &mpi_size);
    MPI_Comm_rank(comm_cart, &mpi_rank);
    MPI_Cart_coords(comm_cart, mpi_rank, 3, mpi_rank_cart.data());

    std::array<int, 3> cmin {0, 0, 0};
    std::array<int, 3> cmax {0, 0, 0};
    for(int i=0; i<3; ++i)
    {
        Nx_local_ng[i] = Nx_glob_ng[i]/mpi_dims_cart[i];
        cmin[i] = Nx_local_ng[i] * mpi_rank_cart[i];

        if(mpi_rank_cart[i]<Nx_glob_ng[i]%mpi_dims_cart[i])
        {
            Nx_local_ng[i]+=1;
            cmin[i] += mpi_rank_cart[i];
        }
        else
        {
            cmin[i] += Nx_glob_ng[i]%mpi_dims_cart[i];
        }

        Nx_local_wg[i] = Nx_local_ng[i] + (2*Nghost[i]);
        cmax[i] = cmin[i] + Nx_local_ng[i];
    }

    range = Range(cmin, cmax, Ng);

    for(int i=0; i<3; ++i)
    {
        is_border[i][0] = (mpi_rank_cart[i] == 0);
        is_border[i][1] = (mpi_rank_cart[i] == mpi_dims_cart[i]-1);
    }

    std::array<int, 3> const remain_dims {0, 1, 1};
    MPI_Cart_sub(comm_cart, remain_dims.data(), &comm_cart_horizontal);
}

void Grid::set_grid(KV_double_1d const& x_glob, KV_double_1d const& y_glob, KV_double_1d const& z_glob)
{
    // Filling x, y, z
    Kokkos::deep_copy(x, Kokkos::subview(x_glob, Kokkos::pair<int, int>(range.Corner_min[0], range.Corner_min[0]+Nx_local_wg[0]+1)));
    Kokkos::deep_copy(y, Kokkos::subview(y_glob, Kokkos::pair<int, int>(range.Corner_min[1], range.Corner_min[1]+Nx_local_wg[1]+1)));
    Kokkos::deep_copy(z, Kokkos::subview(z_glob, Kokkos::pair<int, int>(range.Corner_min[2], range.Corner_min[2]+Nx_local_wg[2]+1)));

    // Filling dx, dy, dz
    compute_cell_size(x, dx);
    compute_cell_size(y, dy);
    compute_cell_size(z, dz);

    // Filling x_center, y_center, z_center
    compute_cell_center(x, x_center);
    compute_cell_center(y, y_center);
    compute_cell_center(z, z_center);

    std::unique_ptr<IComputeGeom> grid_geometry
            = factory_grid_geometry();

    // Filling ds, dv
    grid_geometry->execute(range.all_ghosts(), x, y, z, dx, dy, dz, ds, dv);
}

void Grid::print_grid(std::ostream& os) const
{
    print_info(os, "Ndim", ndim);
    print_info(os, "Nghost[0]", Nghost[0]);
    print_info(os, "Nghost[1]", Nghost[1]);
    print_info(os, "Nghost[2]", Nghost[2]);

    print_info(os, "mpi_size", mpi_size);

    print_info(os, "mpi_dims_cart[0]", mpi_dims_cart[0]);
    print_info(os, "mpi_dims_cart[1]", mpi_dims_cart[1]);
    print_info(os, "mpi_dims_cart[2]", mpi_dims_cart[2]);

    print_info(os, "Nx_glob_ng[0]", Nx_glob_ng[0]);
    print_info(os, "Nx_glob_ng[1]", Nx_glob_ng[1]);
    print_info(os, "Nx_glob_ng[2]", Nx_glob_ng[2]);

    print_info(os, "Nx[0]", Nx_local_ng[0]);
    print_info(os, "Nx[1]", Nx_local_ng[1]);
    print_info(os, "Nx[2]", Nx_local_ng[2]);

    print_info(os, "Nx_local_wg[0]", Nx_local_wg[0]);
    print_info(os, "Nx_local_wg[1]", Nx_local_wg[1]);
    print_info(os, "Nx_local_wg[2]", Nx_local_wg[2]);

    print_info(os, "Corner_min[0]", range.Corner_min[0]);
    print_info(os, "Corner_min[1]", range.Corner_min[1]);
    print_info(os, "Corner_min[2]", range.Corner_min[2]);

    print_info(os, "Corner_max[0]", range.Corner_max[0]);
    print_info(os, "Corner_max[1]", range.Corner_max[1]);
    print_info(os, "Corner_max[2]", range.Corner_max[2]);
}

} // namespace novapp
