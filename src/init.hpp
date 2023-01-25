/**
 * @file init.hpp
 * Initialisation of the data
 */
#pragma once

#include <Kokkos_Core.hpp>

//! Initialisation for the schock tube problem
//! @param[inout] rho density 3D array
//! @param[out] u speed 3D array
//! @param[out] P pressure 3D array
//! @param[in] inter interface position
void ShockTubeInit(
    Kokkos::View<double***> rho,
    Kokkos::View<double***> u,
    Kokkos::View<double***> P,
    int inter);
