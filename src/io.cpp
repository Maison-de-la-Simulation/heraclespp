#include "io.hpp"

void write(int nx, void * rho)
{
    PDI_multi_expose("write_rho_new",
                    "nx", &nx, PDI_OUT,
                    "rho_new", rho, PDI_OUT,
                    NULL);
}