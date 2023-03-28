/**
 * @file grid.cpp
 * Grid class implementation
 */

#include "grid.hpp"
#include "Kokkos_shortcut.hpp"

namespace novapp
{

Grid::Grid(INIReader const& reader)
    : Ng(reader.GetInteger("Grid", "Nghost", 2))
    , Nghost {0, 0, 0}
    , Ncpu_x {1, 1, 1}
    , mpi_rank_cart {0, 0, 0}
{
    Nx_glob_ng[0] = reader.GetInteger("Grid", "Nx_glob", 0); // Cell number
    Nx_glob_ng[1] = reader.GetInteger("Grid", "Ny_glob", 0); // Cell number
    Nx_glob_ng[2] = reader.GetInteger("Grid", "Nz_glob", 0); // Cell number

    for (int idim = 0; idim < Ndim; idim++)
    {
        Nghost[idim] = Ng;
    }

    xmin = reader.GetReal("Grid", "xmin", 0.0);
    xmax = reader.GetReal("Grid", "xmax", 1.0);
    ymin = reader.GetReal("Grid", "ymin", 0.0);
    ymax = reader.GetReal("Grid", "ymax", 1.0);
    zmin = reader.GetReal("Grid", "zmin", 0.0);
    zmax = reader.GetReal("Grid", "zmax", 1.0);

    Lx = xmax - xmin;
    Ly = ymax - ymin;
    Lz = zmax - zmin;

    dx[0] = Lx / Nx_glob_ng[0];
    dx[1] = Ly / Nx_glob_ng[1];
    dx[2] = Lz / Nx_glob_ng[2];

    Ncpu_x[0] = reader.GetInteger("Grid", "Ncpu_x", 0); // number of procs, default 0=>defined by MPI
    Ncpu_x[1] = reader.GetInteger("Grid", "Ncpu_y", 0); // number of procs
    Ncpu_x[2] = reader.GetInteger("Grid", "Ncpu_z", 0); // number of procs

    //!    Type of boundary conditions possibilities are : 
    //!    "Internal", "Periodic", "Reflexive", NullGradient", UserDefined", "Null" (undefined) 
    Nx_local_ng = Nx_glob_ng; // default for a single MPI process

    MPI_Decomp();

    Init_nodes();
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

    MPI_Dims_create(Ncpu, Ndim, Ncpu_x.data());
    for(int n=Ndim; n<3; n++)
    {
        Ncpu_x[n] = 1;
    }
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
    std::array<int, 3> cmin{0, 0, 0};
    std::array<int, 3> cmax{0, 0, 0};
    for(int i=0; i<3; i++)
    {
        remain_dims[i]=true;
        MPI_Cart_sub(comm_cart, remain_dims.data(), &comm_cart_1d[i]);
        MPI_Exscan(&Nx_local_ng[i], &cmin[i], 1, MPI_INT, MPI_SUM, comm_cart_1d[i]);
        cmax[i] = cmin[i] + Nx_local_ng[i];
        
        remain_dims[i]=false;
    }
    range = Range(cmin, cmax, Ng);
    
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

    NBlock[0] = 1; // Default is no sub-block 
    NBlock[1] = 1; // Default is no sub-block 
    NBlock[2] = 1; // Default is no sub-block 

    Nx_block = Nx_local_wg;

    for(int i=0; i<3; i++)
    {
        is_border[i][0] = false;
        is_border[i][1] = false;
    
        if(mpi_rank_cart[i] == 0)           is_border[i][0] = true;
        if(mpi_rank_cart[i] == Ncpu_x[i]-1) is_border[i][1] = true;
    }
}

void Grid::Init_nodes()
{
    x = KDV_double_1d("Initx", Nx_local_ng[0]+ 2*Nghost[0] + 1);
    y = KDV_double_1d("Inity", Nx_local_ng[1]+ 2*Nghost[1] + 1);
    z = KDV_double_1d("Initz", Nx_local_ng[2]+ 2*Nghost[2] + 1);

    offsetx = range.Corner_min[0] - Nghost[0];
    offsety = range.Corner_min[1] - Nghost[1];
    offsetz = range.Corner_min[2] - Nghost[2];

    auto const x_h = x.h_view;
    for (int i = 0; i < Nx_local_ng[0] + 2 * Nghost[0] + 1; ++i)
    {
        x_h(i) = xmin + (i + offsetx) * dx[0]; // Position of the left interface
    }
    x.modify_host();
    x.sync_device();
    auto const y_h = y.h_view;
    for (int i = 0; i < Nx_local_ng[1] + 2 * Nghost[1] + 1; ++i)
    {
        y_h(i) = ymin + (i + offsety) * dx[1]; // Position of the left interface
    }
    y.modify_host();
    y.sync_device();
    auto const z_h = z.h_view;
    for (int i = 0; i < Nx_local_ng[2] + 2 * Nghost[2] + 1; ++i)
    {
        z_h(i) = zmin + (i + offsetz) * dx[2]; // Position of the left interface
    }
    z.modify_host();
    z.sync_device();
}

void Grid::print_grid() const
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

} // namespace novapp
