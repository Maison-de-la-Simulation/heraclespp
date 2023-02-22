/**
 * @file grid.cpp
 * Grid class implementation
 */
#include "grid.hpp"
#include "ndim.hpp"

Grid::Grid(INIReader reader)
{
    Ndim = ndim;
    Ng = reader.GetInteger("Grid", "Nghost", 2); // Ghost cell depth

    Nx_glob_ng[0] = reader.GetInteger("Grid", "Nx_glob", 0); // Cell number
    Nx_glob_ng[1] = reader.GetInteger("Grid", "Ny_glob", 0); // Cell number
    Nx_glob_ng[2] = reader.GetInteger("Grid", "Nz_glob", 0); // Cell number

    Nghost = {0,0,0};
    for (int n=0; n<ndim; n++)
    {
        Nghost[n] = Ng;
    }

    Ncpu_x[0] = reader.GetInteger("Grid", "Ncpu_x", 0); // number of procs, default 0=>defined by MPI
    Ncpu_x[1] = reader.GetInteger("Grid", "Ncpu_y", 1); // number of procs
    Ncpu_x[2] = reader.GetInteger("Grid", "Ncpu_z", 1); // number of procs

    //!    Type of boundary conditions possibilities are : 
    //!    "Internal", "Periodic", "Reflexive", NullGradient", UserDefined", "Null" (undefined) 
    Nx_local_ng      = Nx_glob_ng ; // default for a single MPI process

    MPI_Decomp();
    range.fill_range(Ndim, Nghost);

    grid_type = reader.Get("Grid", "type", "");

    if(grid_type == "rectilinear")
    {
        init_grid_recti();
    }
    else if (grid_type == "irregular rectilinear")
    {
        init_grid_irrecti();
    }
    else
    {
        init_grid_other();
    }
 }


 void Grid::init_grid_recti()
 {
    // additional codes for regular rectiliner grid geometry
 }

void Grid::init_grid_irrecti()
 {
    // additional codes for irregular rectiliner grid geometry
 }

void Grid::init_grid_other()
 {
    // additional codes for other type of grid's geometry
 }

/* ****************************************************************
This routine distribute the cpu over the various direction in an optimum way
Ncpu_x  : Number of cpu along each direction, output
          Ncpu = Ncpu_x[0] * Ncpu_x[1] * Ncpu_x[2]  
*/
void Grid::MPI_Decomp() 
{
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    Ncpu = mpi_size;
    // std::array<int, 3> Ncpu_x_tmp={0,0,0};
    // MPI_Dims_create(Ncpu, Ndim, Ncpu_x_tmp.data());

    // for(int i=0; i<Ndim; i++)
    // {
    //     if(Ncpu_x_tmp[i]>=Nx_glob_ng[i])
    //     {
    //         Ncpu_x[i] = 1+Nx_glob_ng[i]/10 ; // don't decompose if less than 10 cells
    //     }
    // }
    MPI_Dims_create(Ncpu, 3, Ncpu_x.data());

    std::array<int, 3> periodic = {1,1,1};

    MPI_Cart_create(MPI_COMM_WORLD, 3, Ncpu_x.data(), periodic.data(), 0, &comm_cart);
    MPI_Cart_coords(comm_cart, mpi_rank, 3, mpi_rank_cart.data());

    for(int i=0; i<3; i++)
    {
        Nx_local_ng[i] = Nx_glob_ng[i]/Ncpu_x[i];
        if(mpi_rank_cart[i]<Nx_glob_ng[i]%Ncpu_x[i])
        {
            Nx_local_ng[i]+=1;
        }  
        Nx_local_wg[i] = Nx_local_ng[i] + 2*Nghost[i];                         ;
    }
    
    std::array<int, 3> remain_dims= {false, false, false};
    MPI_Comm comm_cart_1d[3];
    for(int i=0; i<3; i++)
    {
        remain_dims[i]=true;
        range.Corner_min[i] = 0;
        MPI_Cart_sub(comm_cart, remain_dims.data(), &comm_cart_1d[i]);
        MPI_Exscan(&Nx_local_ng[i], &range.Corner_min[i], 1, MPI_INT, MPI_SUM, comm_cart_1d[i]);
        range.Corner_max[i] = range.Corner_min[i]+Nx_local_ng[i]-1;
        
        remain_dims[i]=false;
    }
    
    int tmp_coord[3];
    for(int i=-1; i<2; i++)
    {
        for(int j=-1; j<2; j++)
        {
            for(int k=-1; k<2; k++)
            {
                tmp_coord[0] = mpi_rank_cart[0]+i;
                tmp_coord[1] = mpi_rank_cart[1]+j;
                tmp_coord[2] = mpi_rank_cart[2]+k;
                MPI_Cart_rank(comm_cart, tmp_coord, &(NeighborRank[i+1][j+1][k+1]));
            } 
        }
    }  

    NBlock[0] = 1   ; // Default is no sub-block 
    NBlock[1] = 1   ; // Default is no sub-block 
    NBlock[2] = 1   ; // Default is no sub-block 

    Nx_block = Nx_local_wg ;

    for(int i=0; i<3; i++)
    {
        is_border[i][0] = false;
        is_border[i][1] = false;
    
        if(mpi_rank_cart[i] == 0)           is_border[i][0] = true;
        if(mpi_rank_cart[i] == Ncpu_x[i]-1) is_border[i][1] = true;
    }
}

void Grid::print_grid()
{
    if(mpi_rank==0)
    {
        prinf_info("Ndim", Ndim);
        prinf_info("Nghost[0]", Nghost[0]);
        prinf_info("Nghost[1]", Nghost[1]);
        prinf_info("Nghost[2]", Nghost[2]);
        
        prinf_info("Ncpu", Ncpu);

        prinf_info("Ncpu_x[0]", Ncpu_x[0]);
        prinf_info("Ncpu_x[1]", Ncpu_x[1]);
        prinf_info("Ncpu_x[2]", Ncpu_x[2]);

        prinf_info("Nx_glob_ng[0]", Nx_glob_ng[0]);
        prinf_info("Nx_glob_ng[1]", Nx_glob_ng[1]);
        prinf_info("Nx_glob_ng[2]", Nx_glob_ng[2]);

        prinf_info("Nx[0]", Nx_local_ng[0]);
        prinf_info("Nx[1]", Nx_local_ng[1]);
        prinf_info("Nx[2]", Nx_local_ng[2]);

        prinf_info("Nx_local_wg[0]", Nx_local_wg[0]);
        prinf_info("Nx_local_wg[1]", Nx_local_wg[1]);
        prinf_info("Nx_local_wg[2]", Nx_local_wg[2]);

        prinf_info("NBlock[0]", NBlock[0]);
        prinf_info("NBlock[1]", NBlock[1]);
        prinf_info("NBlock[2]", NBlock[2]);

        prinf_info("Nx_block[0]", Nx_block[0]);
        prinf_info("Nx_block[1]", Nx_block[1]);
        prinf_info("Nx_block[2]", Nx_block[2]);

        prinf_info("Corner_min[0]", range.Corner_min[0]);
        prinf_info("Corner_min[1]", range.Corner_min[1]);
        prinf_info("Corner_min[2]", range.Corner_min[2]);

        prinf_info("Corner_max[0]", range.Corner_max[0]);
        prinf_info("Corner_max[1]", range.Corner_max[1]);
        prinf_info("Corner_max[2]", range.Corner_max[2]);
    }
}
