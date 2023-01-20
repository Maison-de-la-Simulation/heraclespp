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

void write(int iter, int nx, void * rho, void *u)
{
    PDI_multi_expose("write_rho_new",
                    "nx", &nx, PDI_OUT,
                    "rho_new", rho, PDI_OUT,
                    "iter", &iter, PDI_OUT,
                    "u", &u, PDI_OUT,
                    NULL);
}
