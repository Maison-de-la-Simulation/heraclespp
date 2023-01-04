/**
 * @file init.hpp
 * Initialisation pour le tube a choc de Sod
 */
#pragma once

#include <Kokkos_Core.hpp>

//! Initialisation for the schock tube problem
//! @param[out] rho density array
//! @param[out] rhou momentum array
//! @param[out] E energy array
//! @param[in] inter interface value
//! @param[in] nx number of cells
void ShockTubeInit(
        Kokkos::View<double*> rho,
        Kokkos::View<double*> rhou,
        Kokkos::View<double*> E,
        int inter,
        int nx);
