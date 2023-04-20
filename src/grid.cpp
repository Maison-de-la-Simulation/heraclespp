/**
 * @file grid.cpp
 * Grid class implementation
 */

#include "grid.hpp"
#include "Kokkos_shortcut.hpp"

namespace novapp
{

Grid::Grid(Param const& param)
    : Nghost {0, 0, 0}
    , Ncpu_x {1, 1, 1}
    , mpi_rank_cart {0, 0, 0}
{
    Ng = param.Ng;

    Nx_glob_ng[0] = param.Nx_glob_ng[0];
    Nx_glob_ng[1] = param.Nx_glob_ng[1];
    Nx_glob_ng[2] = param.Nx_glob_ng[2];

    for (int idim = 0; idim < Ndim; idim++)
    {
        Nghost[idim] = Ng;
    }

    xmin = param.xmin;
    xmax = param.xmax;
    ymin = param.ymin;
    ymax = param.ymax;
    zmin = param.zmin;
    zmax = param.zmax;

    L[0] = xmax - xmin;
    L[1] = ymax - ymin;
    L[2] = zmax - zmin;

    Ncpu_x[0] = param.Ncpu_x[0];
    Ncpu_x[1] = param.Ncpu_x[1];
    Ncpu_x[2] = param.Ncpu_x[2];

    //!    Type of boundary conditions possibilities are : 
    //!    "Internal", "Periodic", "Reflexive", NullGradient", UserDefined", "Null" (undefined) 
    Nx_local_ng = param.Nx_glob_ng; // default for a single MPI process

    MPI_Decomp();
    Init_nodes();
    Init_grid();
}

/* ****************************************************************
This routine distribute the cpu over the various direction in an optimum way
Ncpu_x  : Number of cpu along each direction, output
          Ncpu = Ncpu_x[0] * Ncpu_x[1] * Ncpu_x[2]  
*/
void Grid::MPI_Decomp() 
{
    MPI_Comm_size(MPI_COMM_WORLD, &Ncpu);

    MPI_Dims_create(Ncpu, Ndim, Ncpu_x.data());
    for(int n=Ndim; n<3; n++)
    {
        Ncpu_x[n] = 1;
    }
    std::array<int, 3> periodic = {1,1,1};

    MPI_Cart_create(MPI_COMM_WORLD, 3, Ncpu_x.data(), periodic.data(), 1, &comm_cart);
    MPI_Comm_size(comm_cart, &mpi_size);
    MPI_Comm_rank(comm_cart, &mpi_rank);
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
    std::array<int, 3> cmin{0, 0, 0};
    std::array<int, 3> cmax{0, 0, 0};
    for(int i=0; i<3; i++)
    {
        remain_dims[i]=true;
        MPI_Comm comm_cart_1d;
        MPI_Cart_sub(comm_cart, remain_dims.data(), &comm_cart_1d);
        MPI_Exscan(&Nx_local_ng[i], &cmin[i], 1, MPI_INT, MPI_SUM, comm_cart_1d);
        MPI_Comm_free(&comm_cart_1d);
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
    for(int idim=0; idim<ndim; idim++)
    {
        for(int iface=0; iface<2; iface++)
        {
            int displ = iface==0? -1 : 1;
            MPI_Cart_shift(comm_cart, idim, displ, &neighbor_src[idim*2+iface], &neighbor_dest[idim*2+iface]);
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
    x = KDV_double_1d("Initx", Nx_local_ng[0]+2*Nghost[0]+1);
    y = KDV_double_1d("Inity", Nx_local_ng[1]+2*Nghost[1]+1);
    z = KDV_double_1d("Initz", Nx_local_ng[2]+2*Nghost[2]+1);

    offsetx = range.Corner_min[0] - Nghost[0];
    offsety = range.Corner_min[1] - Nghost[1];
    offsetz = range.Corner_min[2] - Nghost[2];

    double dx = L[0] / Nx_glob_ng[0];
    double dy = L[1] / Nx_glob_ng[1];
    double dz = L[2] / Nx_glob_ng[2];

    auto const x_h = x.h_view;
    for (int i=0; i<Nx_local_ng[0]+2*Nghost[0]+1; ++i)
    {
        x_h(i) = xmin + (i + offsetx) * dx; // Position of the left interface
    }
    x.modify_host();
    x.sync_device();
    auto const y_h = y.h_view;
    for (int i=0; i<Nx_local_ng[1]+2*Nghost[1]+1; ++i)
    {
        y_h(i) = ymin + (i + offsety) * dy; // Position of the left interface
    }
    y.modify_host();
    y.sync_device();
    auto const z_h = z.h_view;
    for (int i=0; i<Nx_local_ng[2]+2*Nghost[2]+1; ++i)
    {
        z_h(i) = zmin + (i + offsetz) * dz; // Position of the left interface
    }
    z.modify_host();
    z.sync_device();
}

void Grid::Init_grid()
{
    dx = KDV_double_4d("dx", Nx_local_ng[0]+2*Nghost[0],
                             Nx_local_ng[1]+2*Nghost[1],
                             Nx_local_ng[2]+2*Nghost[2],
                             3);
    ds = KV_double_4d("ds", Nx_local_ng[0]+2*Nghost[0], 
                             Nx_local_ng[1]+2*Nghost[1],
                             Nx_local_ng[2]+2*Nghost[2],
                             3);
    dv = KV_double_3d("dv", Nx_local_ng[0]+2*Nghost[0], 
                             Nx_local_ng[1]+2*Nghost[1],
                             Nx_local_ng[2]+2*Nghost[2]);
    dx_inter = KV_double_1d("dx_inter", 3);

    auto const dx_d = dx.d_view;
    auto const x_d = x.d_view;
    auto const y_d = y.d_view;
    auto const z_d = z.d_view;
    Kokkos::parallel_for("File_dx",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, 
                            {Nx_local_ng[0]+2*Nghost[0], 
                             Nx_local_ng[1]+2*Nghost[1],
                             Nx_local_ng[2]+2*Nghost[2]}),
    KOKKOS_CLASS_LAMBDA(int i, int j, int k)
    {
        dx_d(i, j, k, 0) = x_d(i+1) - x_d(i);
        dx_d(i, j, k, 1) = y_d(j+1) - y_d(j);
        dx_d(i, j, k, 2) = z_d(k+1) - z_d(k);
    });
    dx.modify_device();
    dx.sync_host();

    Kokkos::parallel_for("File_ds",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, 
                            {Nx_local_ng[0]+2*Nghost[0], 
                             Nx_local_ng[1]+2*Nghost[1],
                             Nx_local_ng[2]+2*Nghost[2]}),
    KOKKOS_CLASS_LAMBDA(int i, int j, int k)
    {
        for (int idim=0; idim<3; ++idim)
        {
            dx_inter[0] = dx_d(i, j, k, 0);
            dx_inter[1] = dx_d(i, j, k, 1);
            dx_inter[2] = dx_d(i, j, k, 2);
            dx_inter[idim] = 1;
            ds(i, j, k, idim) = dx_inter[0] * dx_inter[1] * dx_inter[2];
        }
    });

    Kokkos::parallel_for("File_dv",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, 
                            {Nx_local_ng[0]+2*Nghost[0], 
                             Nx_local_ng[1]+2*Nghost[1],
                             Nx_local_ng[2]+2*Nghost[2]}),
    KOKKOS_CLASS_LAMBDA(int i, int j, int k)
    {
        dv(i, j, k) = 1;
        for (int idim=0; idim<ndim; ++idim)
        {
            dv(i, j, k) *= dx_d(i, j, k, idim);
        }
    });
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
