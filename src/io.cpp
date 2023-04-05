#include "io.hpp"

#include <pdi.h>
#include <mpi.h>

namespace novapp
{

void write_pdi_init(int max_iter, int frequency, Grid const& grid)
{
    int mpi_rank = grid.mpi_rank;
    int mpi_size = grid.mpi_size;
    int simu_ndim = grid.Ndim;

    PDI_multi_expose("init_PDI",
                    "max_iter", &max_iter, PDI_OUT,
                    "frequency", &frequency, PDI_OUT,
                    "ndim", &simu_ndim, PDI_OUT,
                    "mpi_rank", &mpi_rank, PDI_OUT,
                    "mpi_size", &mpi_size, PDI_OUT,
                    "n_ghost", grid.Nghost.data(), PDI_OUT,
                    "nx_glob_ng", grid.Nx_glob_ng.data(), PDI_OUT,
                    "nx_local_ng", grid.Nx_local_ng.data(), PDI_OUT,
                    "nx_local_wg", grid.Nx_local_wg.data(), PDI_OUT,
                    "start", grid.range.Corner_min.data(), PDI_OUT,
                    NULL);
}

void write_pdi(int iter, double t, KDV_double_3d rho,
                                   KDV_double_4d u,
                                   KDV_double_3d P, 
                                   KDV_double_3d E,
                                   KDV_double_1d x,
                                   KDV_double_1d y,
                                   KDV_double_1d z)
{
    rho.sync_host();
    u.sync_host();
    P.sync_host();
    E.sync_host();
    PDI_multi_expose("write_file",
                    "iter", &iter, PDI_OUT,
                    "current_time", &t, PDI_OUT, 
                    "rho", rho.h_view.data(), PDI_OUT,
                    "u", u.h_view.data(), PDI_OUT,
                    "P", P.h_view.data(), PDI_OUT,
                    "E", E.h_view.data(), PDI_OUT,
                    "x", x.h_view.data(), PDI_OUT,
                    "y", y.h_view.data(), PDI_OUT,
                    "z", z.h_view.data(), PDI_OUT,
                    NULL);
}

bool should_output(int iter, int freq, int iter_max, double current, double dt, double time_out)
{
    bool result = (freq > 0) && (((iter+1)>=iter_max) || ((iter+1)%freq==0) || (current+dt>=time_out));
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    if(result && (mpi_rank==0))
    {
        std::cout<< std::left << std::setw(80) << std::setfill('*') << "*"<<std::endl;
        std::cout<<"current iteration "<<iter+1<<" : "<<std::endl;
        std::cout<<"current time = "<<current<<" ( ~ "<< 100*(current)/time_out <<"%)"<<std::endl<<std::endl ;
    }

    return result;
}

void read_pdi(std::string restart_file,
              KDV_double_3d rho,
              KDV_double_4d u,
              KDV_double_3d P,
              double &t, int &iter)
{
    int filename_size = restart_file.size();
    PDI_multi_expose("read_file",
                    "restart_filename_size", &filename_size, PDI_INOUT,
                    "restart_filename", restart_file.data(), PDI_INOUT,
                    "iter", &iter, PDI_INOUT,
                    "current_time", &t, PDI_INOUT, 
                    "rho", rho.h_view.data(), PDI_INOUT,
                    "u", u.h_view.data(), PDI_INOUT,
                    "P", P.h_view.data(), PDI_INOUT,
                    NULL);
    rho.modify_host();
    u.modify_host();
    P.modify_host();
}

} // namespace novapp
