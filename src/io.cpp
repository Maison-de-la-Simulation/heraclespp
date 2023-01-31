#include "io.hpp"

#include <pdi.h>

void init_write(int max_iter, int frequency, int ghost)
{
    PDI_multi_expose("init_PDI",
                    "max_iter", &max_iter, PDI_OUT,
                    "frequency", &frequency, PDI_OUT,
                    "ghost_depth", &ghost, PDI_OUT,
                    NULL);
}

void write(int iter, int nx, double current, void * rho, void *u, void *P)
{
    PDI_multi_expose("write_file",
                    "nx", &nx, PDI_OUT,
                    "current_time", &current, PDI_OUT,
                    "iter", &iter, PDI_OUT,
                    "rho", rho, PDI_OUT,
                    "u", u, PDI_OUT,
                    "P", P, PDI_OUT,
                    NULL);
}

bool should_output(int iter, int freq, int iter_max, double current, double dt, double time_out)
{
    bool result = (iter>=iter_max) || (iter%freq==0) || (current+dt>=time_out);
    return result;
}
