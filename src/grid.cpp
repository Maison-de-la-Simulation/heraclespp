/**
 * @file grid.cpp
 * Grid class implementation
 */

#include "grid.hpp"

Grid::Grid(INIReader reader)
 {
    Ndim     = 3 ; 
    Nghost   = 2 ; 
    
    Nx_glob_ng[0] = reader.GetInteger("Grid", "Nx_glob", 0); // Cell number
    Nx_glob_ng[1] = reader.GetInteger("Grid", "Ny_glob", 0); // Cell number
    Nx_glob_ng[2] = reader.GetInteger("Grid", "Nz_glob", 0); // Cell number


    //!    Type of boundary conditions possibilities are : 
    //!    "Internal", "Periodic", "Reflexive", NullGradient", UserDefined", "Null" (undefined) 
    Nx_local_ng      = Nx_glob_ng ; // default for a single MPI process

    MPI_Decomp();

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
 } ;


 void Grid::init_grid_recti()
 {
    // std::cout<<"init_grid_recti"<<std::endl;
    grid_coords_name[0] = "coord_X";
    grid_coords_name[1] = "coord_Y";
    grid_coords_name[2] = "coord_Z";
 };

void Grid::init_grid_irrecti()
 {
    // std::cout<<"init_grid_irrecti"<<std::endl;
    grid_coords_name[0] = "coord_X";
    grid_coords_name[1] = "coord_Y";
    grid_coords_name[2] = "coord_Z";
 };

void Grid::init_grid_other()
 {
    // std::cout<<"init_grid_other"<<std::endl;
    grid_coords_name[0] = "coord_R";
    grid_coords_name[1] = "coord_theta";
    grid_coords_name[2] = "coord_Z";
 };

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
    std::array<int, 3> Ncpu_x_tmp={0,0,0};
    MPI_Dims_create(Ncpu, Ndim, Ncpu_x_tmp.data());
    
    for(int i=0; i<Ndim; i++)
    {
        Ncpu_x[i] = 0;
        if(Ncpu_x_tmp[i]>=Nx_glob_ng[i])
        {
            Ncpu_x[i] = Nx_glob_ng[i];
        }
    }
    
    MPI_Dims_create(Ncpu, Ndim, Ncpu_x.data());

    std::array<int, 3> periodic = {1,1,1} ;

    MPI_Cart_create(MPI_COMM_WORLD, Ndim, Ncpu_x.data(), periodic.data(), 1, &comm_cart);
    MPI_Cart_coords(comm_cart, mpi_rank, Ndim, mpi_rank_cart.data());

    for(int i=0; i<Ndim; i++)
    {
        Nx_local_ng[i] = Nx_glob_ng[i]/Ncpu_x[i];
        if(mpi_rank_cart[i]<Nx_glob_ng[i]%Ncpu_x[i])
        {
            Nx_local_ng[i]+=1;
        }  
        Nx_local_wg[i] = Nx_local_ng[i] + 2*Nghost;                         ;
    }
    
    for(int i=0; i<Ndim; i++)
    {
        cornerPosition[i][0] = mpi_rank_cart[i]*Nx_glob_ng[i]/Ncpu_x[i];
        if(mpi_rank_cart[i]>=Nx_glob_ng[i]%Ncpu_x[i])
        {
            cornerPosition[i][0]+=1;
        }
        cornerPosition[i][1] = cornerPosition[i][0] + Nx_local_ng[i]-1;
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
}

void Grid::print_grid()
{
    int mpiRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

    if(mpiRank==0)
    {
        prinf_info("Ndim", Ndim);
        prinf_info("Nghost", Nghost);
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

        prinf_info("Corner_min[0]", cornerPosition[0][0]);
        prinf_info("Corner_min[1]", cornerPosition[1][0]);
        prinf_info("Corner_min[2]", cornerPosition[2][0]);

        prinf_info("Corner_max[0]", cornerPosition[0][1]);
        prinf_info("Corner_max[1]", cornerPosition[1][1]);
        prinf_info("Corner_max[2]", cornerPosition[2][1]);
    }
};