
/**
 * @file grid.hpp
 * Grid class declaration
 */
#pragma once
#include <iostream> 
#include <array>
#include <string>
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <inih/INIReader.hpp>
#include <mpi.h>
#include <iomanip>


template<class T>
inline
void prinf_info(std::string var_name, T var_value)
{
    using namespace std;
    cout << std::left << std::setw(40) << std::setfill('.') << var_name;
    cout << std::right << std::setw(40) << std::setfill('.') << var_value<<endl;
};

class Grid
{
public :
    Grid(INIReader reader);
    void print_grid();

private:
    void MPI_Decomp();
    // void Domain_partitioning();
    void domainDecomp();
    void init_grid_recti();
    void init_grid_irrecti();
    void init_grid_other();

    
  
public :
    int Ndim; // Number of dimension of the run 1-3 (default = 1)
    int Nghost; // Number of ghost cells (default = 2)

    std::array<int, 3> Nx_glob_ng;    // Total number of cells in each directions (excluding ghost)
    std::array<int, 3> Nx_local_ng;    // Number of cells on the local MPI process (excluding ghost)
    std::array<int, 3> Nx_local_wg;    // Number of cells on the local MPI process (excluding ghost)
    // std::array<int, 3> Nx_size;    // Size of local arrays (including ghost cells) 
    std::array<int, 3> NBlock;     // number of sub-blocks (default (1,1,1))
    std::array<int, 3> Nx_block;   // Maximum size of sub-block, including ghost
    std::array<std::array<int, 2>,3> cornerPosition; // Position of corner in the global grid (no ghost)

    std::array<std::array<std::array<int, 3>, 3>, 3> NeighborRank;
    std::array<int,3> Ncpu_x;
    int Ncpu;

    MPI_Comm comm_cart;
    int mpi_rank;
    int mpi_size;
    std::array<int, 3> mpi_rank_cart;

private:
    std::string grid_type;
    std::string grid_coords_name[3];    
};