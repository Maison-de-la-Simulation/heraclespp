/**
 * @file grid.cpp
 * Grid class implementation
 */

#include "grid.hpp"
#include "kokkos_shortcut.hpp"
#include "factories.hpp"

namespace
{

template<class T>
inline
void print_info(std::string var_name, T var_value)
{
    using namespace std;
    cout << left  << setw(40) << setfill('.') << var_name;
    cout << right << setw(40) << setfill('.') << var_value << endl;
}

} // namespace

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
    
    Ncpu_x[0] = param.Ncpu_x[0];
    Ncpu_x[1] = param.Ncpu_x[1];
    Ncpu_x[2] = param.Ncpu_x[2];

    //!    Type of boundary conditions possibilities are : 
    //!    "Internal", "Periodic", "Reflexive", NullGradient", UserDefined", "Null" (undefined) 
    Nx_local_ng = param.Nx_glob_ng; // default for a single MPI process

    MPI_Decomp();
    Init_grid(param);
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
        start_cell_wg[i] = Nx_local_ng[i] * mpi_rank_cart[i];

        if(mpi_rank_cart[i]<Nx_glob_ng[i]%Ncpu_x[i])
        {
            Nx_local_ng[i]+=1;
            start_cell_wg[i] += mpi_rank_cart[i]; 
        } 
        else
        {
            start_cell_wg[i] += Nx_glob_ng[i]%Ncpu_x[i];
        } 
        Nx_local_wg[i] = Nx_local_ng[i] + 2*Nghost[i];
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

void Grid::Init_grid(Param const& param)
{
    std::unique_ptr<IGridType> grid_type
            = factory_grid_type(param.grid_type, param);
    
    x = KDV_double_1d("x", Nx_local_wg[0]+1);
    y = KDV_double_1d("y", Nx_local_wg[1]+1);
    z = KDV_double_1d("z", Nx_local_wg[2]+1);

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
    dx_inter = KV_double_1d("dx_inter", 3);
    
    x_glob = Kokkos::View<double*, Kokkos::HostSpace>("x_glob", Nx_glob_ng[0]+2*Nghost[0]+1);
    y_glob = Kokkos::View<double*, Kokkos::HostSpace>("y_glob", Nx_glob_ng[1]+2*Nghost[1]+1);
    z_glob = Kokkos::View<double*, Kokkos::HostSpace>("z_glob", Nx_glob_ng[2]+2*Nghost[2]+1);

    Kokkos::deep_copy(Kokkos::subview(x_glob, Kokkos::pair<int, int>(start_cell_wg[0], start_cell_wg[0]+Nx_local_wg[0]+1)), x.h_view);
    Kokkos::deep_copy(Kokkos::subview(y_glob, Kokkos::pair<int, int>(start_cell_wg[1], start_cell_wg[1]+Nx_local_wg[1]+1)), y.h_view);
    Kokkos::deep_copy(Kokkos::subview(z_glob, Kokkos::pair<int, int>(start_cell_wg[2], start_cell_wg[2]+Nx_local_wg[2]+1)), z.h_view);

    grid_type->execute(x_glob, y_glob, z_glob, Nghost, Nx_local_wg, Nx_glob_ng);

    Kokkos::deep_copy(x.h_view, Kokkos::subview(x_glob, Kokkos::pair<int, int>(start_cell_wg[0], start_cell_wg[0]+Nx_local_wg[0]+1)));
    Kokkos::deep_copy(y.h_view, Kokkos::subview(y_glob, Kokkos::pair<int, int>(start_cell_wg[1], start_cell_wg[1]+Nx_local_wg[1]+1)));
    Kokkos::deep_copy(z.h_view, Kokkos::subview(z_glob, Kokkos::pair<int, int>(start_cell_wg[2], start_cell_wg[2]+Nx_local_wg[2]+1)));

    x.modify_host();
    x.sync_device();
    y.modify_host();
    y.sync_device();
    z.modify_host();
    z.sync_device();

    auto const x_d = x.d_view;
    auto const y_d = y.d_view;
    auto const z_d = z.d_view;

    Kokkos::parallel_for("set_dx_xcenter",
    Nx_local_wg[0],
    KOKKOS_CLASS_LAMBDA(int i)
    {
        dx(i) = x_d(i+1) - x_d(i);
        x_center(i) = x_d(i) + dx(i) / 2;
    });

    Kokkos::parallel_for("set_dy_ycenter",
    Nx_local_wg[1],
    KOKKOS_CLASS_LAMBDA(int i)
    {
        dy(i) = y_d(i+1) - y_d(i);
        y_center(i) = y_d(i) + dy(i) / 2;
    });

    Kokkos::parallel_for("set_dz_zcenter",
    Nx_local_wg[2],
    KOKKOS_CLASS_LAMBDA(int i)
    {
        dz(i) = z_d(i+1) - z_d(i);
        z_center(i) = z_d(i) + dz(i) / 2;
    });

    Kokkos::parallel_for("File_ds",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, 
                            {Nx_local_wg[0], 
                             Nx_local_wg[1],
                             Nx_local_wg[2]}),
    KOKKOS_CLASS_LAMBDA(int i, int j, int k)
    {
        for (int idim=0; idim<3; ++idim)
        {
            dx_inter[0] = dx(i);
            dx_inter[1] = dy(j);
            dx_inter[2] = dz(k);
            dx_inter[idim] = 1;
            ds(i, j, k, idim) = dx_inter[0] * dx_inter[1] * dx_inter[2];
        }
    });

    Kokkos::parallel_for("File_dv",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, 
                            {Nx_local_wg[0], 
                             Nx_local_wg[1],
                             Nx_local_wg[2]}),
    KOKKOS_CLASS_LAMBDA(int i, int j, int k)
    {
        dv(i, j, k) = dx(i) * dy(j) * dz(k);
    });
}

void Grid::print_grid() const
{
    if(mpi_rank==0)
    {
        print_info("Ndim", Ndim);
        print_info("Nghost[0]", Nghost[0]);
        print_info("Nghost[1]", Nghost[1]);
        print_info("Nghost[2]", Nghost[2]);
        
        print_info("Ncpu", Ncpu);

        print_info("Ncpu_x[0]", Ncpu_x[0]);
        print_info("Ncpu_x[1]", Ncpu_x[1]);
        print_info("Ncpu_x[2]", Ncpu_x[2]);

        print_info("Nx_glob_ng[0]", Nx_glob_ng[0]);
        print_info("Nx_glob_ng[1]", Nx_glob_ng[1]);
        print_info("Nx_glob_ng[2]", Nx_glob_ng[2]);

        print_info("Nx[0]", Nx_local_ng[0]);
        print_info("Nx[1]", Nx_local_ng[1]);
        print_info("Nx[2]", Nx_local_ng[2]);

        print_info("Nx_local_wg[0]", Nx_local_wg[0]);
        print_info("Nx_local_wg[1]", Nx_local_wg[1]);
        print_info("Nx_local_wg[2]", Nx_local_wg[2]);

        print_info("NBlock[0]", NBlock[0]);
        print_info("NBlock[1]", NBlock[1]);
        print_info("NBlock[2]", NBlock[2]);

        print_info("Nx_block[0]", Nx_block[0]);
        print_info("Nx_block[1]", Nx_block[1]);
        print_info("Nx_block[2]", Nx_block[2]);

        print_info("Corner_min[0]", range.Corner_min[0]);
        print_info("Corner_min[1]", range.Corner_min[1]);
        print_info("Corner_min[2]", range.Corner_min[2]);

        print_info("Corner_max[0]", range.Corner_max[0]);
        print_info("Corner_max[1]", range.Corner_max[1]);
        print_info("Corner_max[2]", range.Corner_max[2]);
    }
}

} // namespace novapp
