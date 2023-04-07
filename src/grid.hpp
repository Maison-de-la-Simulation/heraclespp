//!
//! @file grid.hpp
//! Grid class declaration
//!

#pragma once

#include <iostream>
#include <array>
#include <string>
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <inih/INIReader.hpp>
#include "Kokkos_shortcut.hpp"
#include <mpi.h>
#include <iomanip>
#include "range.hpp"
#include "ndim.hpp"
#include "Kokkos_shortcut.hpp"

namespace novapp
{

template<class T>
inline
void prinf_info(std::string var_name, T var_value)
{
    using namespace std;
    cout << left  << setw(40) << setfill('.') << var_name;
    cout << right << setw(40) << setfill('.') << var_value << endl;
}

class Grid
{
public :
    static constexpr int Ndim = ndim;

    explicit Grid(INIReader const& reader);
    void print_grid() const;

private:
    void MPI_Decomp();
    void Init_nodes();

public :
    int Ng;
    std::array<int, 3> Nghost;    // Number of ghost cells in each direction (default is 2)
    std::array<int, 3> Nx_glob_ng;    // Total number of cells in each directions (excluding ghost)
    std::array<int, 3> Nx_local_ng;    // Number of cells on the local MPI process (excluding ghost)
    std::array<int, 3> Nx_local_wg;    // Number of cells on the local MPI process (including ghost)
    std::array<int, 3> NBlock;     // number of sub-blocks (default (1,1,1))
    std::array<int, 3> Nx_block;   // Maximum size of sub-block, including ghos
    
    double xmin;
    double xmax;
    double ymin;
    double ymax;
    double zmin;
    double zmax;

    Kokkos::Array<double, 3> L;

    Kokkos::Array<double, 3> dx;

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

    int offsetx;
    int offsety;
    int offsetz;

    KDV_double_1d x;
    KDV_double_1d y;
    KDV_double_1d z;

    std::array<KDV_double_4d, ndim> mpi_buffer;
};

} // namespace novapp
