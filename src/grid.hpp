//!
//! @file grid.hpp
//! Grid class declaration
//!

#pragma once

#include <iostream>
#include <array>
#include <string_view>
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <inih/INIReader.hpp>
#include <mpi.h>
#include <iomanip>
#include "range.hpp"
#include "ndim.hpp"
#include "kokkos_shortcut.hpp"
#include "nova_params.hpp"
#include "grid_type.hpp"

namespace novapp
{

template <class T>
void print_info(std::string_view const var_name, T const var_value)
{
    using namespace std;
    cout << left  << setw(40) << setfill('.') << var_name;
    cout << right << setw(40) << setfill('.') << var_value << endl;
}

class Grid
{
public :
    static constexpr int Ndim = ndim;

    explicit Grid(Param const& param);
    void Init_grid(Param const& param);
    void print_grid() const;
    
private:
    void MPI_Decomp();

public :
    int Ng;
    std::array<int, 3> Nghost;    // Number of ghost cells in each direction (default is 2)
    std::array<int, 3> Nx_glob_ng;    // Total number of cells in each directions (excluding ghost)
    std::array<int, 3> Nx_local_ng;    // Number of cells on the local MPI process (excluding ghost)
    std::array<int, 3> Nx_local_wg;    // Number of cells on the local MPI process (including ghost)
    std::array<int, 3> NBlock;     // number of sub-blocks (default (1,1,1))
    std::array<int, 3> Nx_block;   // Maximum size of sub-block, including ghos
    std::array<int, 3> start_cell_wg; // for local MPI process, index of starting cell wrt the global index with ghost

    std::array<std::array<std::array<int, 3>, 3>, 3> NeighborRank;
    std::array<int, Ndim*2> neighbor_src;
    std::array<int, Ndim*2> neighbor_dest;
    std::array<int,3> Ncpu_x;
    int Ncpu;

    MPI_Comm comm_cart;
    int mpi_rank;
    int mpi_size;
    std::array<int, 3> mpi_rank_cart;

    Range range;
    std::array<std::array<bool, 2>,3> is_border;

    Kokkos::View<double*, Kokkos::HostSpace> x_glob;
    Kokkos::View<double*, Kokkos::HostSpace> y_glob;
    Kokkos::View<double*, Kokkos::HostSpace> z_glob;

    KDV_double_1d x;
    KDV_double_1d y;
    KDV_double_1d z;

    KV_double_1d x_center;
    KV_double_1d y_center;
    KV_double_1d z_center;

    KV_double_1d dx;
    KV_double_1d dy;
    KV_double_1d dz;

    KV_double_4d ds;
    KV_double_3d dv;
};

} // namespace novapp
