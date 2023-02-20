/**
 * @file buffer.hpp
 * buffer class declaration
 */

#pragma once

#include <cmath>
#include <array>
#include <string>
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <mpi.h>
#include "buffer.hpp"

void innerExchangeMPI(Kokkos::View<double***> rho,
                      Kokkos::View<double****> rhou,
                      Kokkos::View<double***> E, 
                      int *ng, MPI_Comm *comm_cart, std::array<std::array<std::array<int, 3>, 3>, 3> nrank,
                      Buffer *send_buffer, Buffer* recv_buffer);
