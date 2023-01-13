#include "io.hpp"

void init_write(int max_iter, int frequency)
{
    PDI_multi_expose("init_PDI",
                    "max_iter", &max_iter, PDI_OUT,
                    "frequency", &frequency, PDI_OUT,
                    NULL);
}

void write(int iter, int nx, void * rho)
{
    PDI_multi_expose("write_rho_new",
                    "nx", &nx, PDI_OUT,
                    "rho_new", rho, PDI_OUT,
                    "iter", &iter, PDI_OUT,
                    NULL);
}