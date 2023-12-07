//!
//! @file grid.hpp
//! Grid class declaration
//!

#pragma once

#include <mpi.h>

#include <array>
#include <iomanip>
#include <iostream>
#include <string_view>

#include <kokkos_shortcut.hpp>
#include <ndim.hpp>

#include "range.hpp"

namespace novapp
{

class Param;

template <class T>
void print_info(std::string_view const var_name, T const var_value)
{
    using namespace std;
    cout << left  << setw(40) << setfill('.') << var_name;
    cout << right << setw(40) << setfill('.') << var_value << endl;
}

class Grid
{
public:
    int Ng;
    std::array<int, 3> Nghost;    // Number of ghost cells in each direction (default is 2)
    std::array<int, 3> Nx_glob_ng;    // Total number of cells in each directions (excluding ghost)
    std::array<int, 3> Nx_local_ng;    // Number of cells on the local MPI process (excluding ghost)
    std::array<int, 3> Nx_local_wg;    // Number of cells on the local MPI process (including ghost)
    std::array<int, 3> NBlock;     // number of sub-blocks (default (1,1,1))
    std::array<int, 3> Nx_block;   // Maximum size of sub-block, including ghos
    std::array<int, 3> start_cell_wg; // for local MPI process, index of starting cell wrt the global index with ghost

    std::array<std::array<std::array<int, 3>, 3>, 3> NeighborRank;
    std::array<int, ndim*2> neighbor_src;
    std::array<int, ndim*2> neighbor_dest;
    std::array<int,3> Ncpu_x;
    int Ncpu;

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

    void print_grid() const;

    void set_grid(KV_double_1d const& x_glob, KV_double_1d const& y_glob, KV_double_1d const& z_glob);

private:
    void MPI_Decomp();
};

template <class... Args>
auto allocate_double_with_ghosts(std::string const& label, Grid const& grid, Args... args)
{
    if constexpr (sizeof...(Args) == 0) {
        return KV_double_3d(
                label,
                grid.Nx_local_wg[0],
                grid.Nx_local_wg[1],
                grid.Nx_local_wg[2],
                args...);
    }
    if constexpr (sizeof...(Args) == 1) {
        return KV_double_4d(
                label,
                grid.Nx_local_wg[0],
                grid.Nx_local_wg[1],
                grid.Nx_local_wg[2],
                args...);
    }
    if constexpr (sizeof...(Args) == 2) {
        return KV_double_5d(
                label,
                grid.Nx_local_wg[0],
                grid.Nx_local_wg[1],
                grid.Nx_local_wg[2],
                args...);
    }
    if constexpr (sizeof...(Args) == 3) {
        return KV_double_6d(
                label,
                grid.Nx_local_wg[0],
                grid.Nx_local_wg[1],
                grid.Nx_local_wg[2],
                args...);
    }
}

} // namespace novapp
