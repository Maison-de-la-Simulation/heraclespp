#include <mpi.h>
#include "mpi_fonc.hpp"

void init_mpi()
{
    int mpi_err;
    mpi_err = MPI_Init(NULL, NULL);
    if(mpi_err!=MPI_SUCCESS)
    {
        std::cout<<"Error in MPI_Init(NULL, NULL)"<<std::endl;
    }
}

void finalize_mpi()
{
    int mpi_err;
    mpi_err = MPI_Finalize();
    if(mpi_err!=MPI_SUCCESS)
    {
        std::cout<<"Error in MPI_Finalize()"<<std::endl;
    }
}
