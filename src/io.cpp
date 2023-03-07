#include "io.hpp"

#include <pdi.h>
#include <mpi.h>

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

void write_pdi(int iter, double t, void * rho, void *u, void *P)
{
    PDI_multi_expose("write_file",
                    "iter", &iter, PDI_OUT,
                    "current_time", &t, PDI_OUT, 
                    "rho", rho, PDI_OUT,
                    "u", u, PDI_OUT,
                    "P", P, PDI_OUT,
                    NULL);
}

bool should_output(int iter, int freq, int iter_max, double current, double dt, double time_out)
{
    return ((iter+1)>=iter_max) || ((iter+1)%freq==0) || (current+dt>=time_out);
}
