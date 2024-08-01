//!
//! @file grid.hpp
//! Grid class declaration
//!

#pragma once

#include <mpi.h>

#include <array>
#include <iosfwd>

#include <kokkos_shortcut.hpp>
#include <ndim.hpp>

#include "range.hpp"

namespace novapp
{

class Param;

class Grid
{
public:
    int Ng;
    std::array<int, 3> Nghost;    // Number of ghost cells in each direction (default is 2)
    std::array<int, 3> Nx_glob_ng;    // Total number of cells in each directions (excluding ghost)
    std::array<int, 3> Nx_local_ng;    // Number of cells on the local MPI process (excluding ghost)
    std::array<int, 3> Nx_local_wg;    // Number of cells on the local MPI process (including ghost)

    std::array<int,3> Ncpu_x;

    MPI_Comm comm_cart;
    int mpi_rank;
    int mpi_size;
    std::array<int, 3> mpi_rank_cart;

    Range range;
    std::array<std::array<bool, 2>,3> is_border;

    KV_double_1d x;
    KV_double_1d y;
    KV_double_1d z;

    KV_double_1d x_center;
    KV_double_1d y_center;
    KV_double_1d z_center;

    KV_double_1d dx;
    KV_double_1d dy;
    KV_double_1d dz;

    KV_double_4d ds;
    KV_double_3d dv;

    explicit Grid(Param const& param);

    Grid(Grid const& rhs) = delete;

    Grid(Grid&& rhs) noexcept = delete;

    ~Grid() noexcept;

    Grid& operator=(Grid const& rhs) = delete;

    Grid& operator=(Grid&& rhs) noexcept = delete;

    void print_grid(std::ostream& os) const;

    void set_grid(KV_double_1d const& x_glob, KV_double_1d const& y_glob, KV_double_1d const& z_glob);

private:
    void MPI_Decomp();
};

} // namespace novapp
