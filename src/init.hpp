/**
 * @file init.hpp
 * Initialisation of the data
 */
#pragma once

#include <Kokkos_Core.hpp>

//! Initialisation for the schock tube problem
//! @param[inout] rho density array
//! @param[out] u speed array
//! @param[out] P pressue array
//! @param[in] inter interface value
//! @param[in] nx number of cells
void ShockTubeInit(
        Kokkos::View<double*> rho,
        Kokkos::View<double*> u,
        Kokkos::View<double*> P,
        int inter,
        int nx);
