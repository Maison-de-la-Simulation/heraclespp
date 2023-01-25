/**
 * @file array_conversion.hpp
 * General fonctions
 */
#pragma once

#include <Kokkos_Core.hpp>

//! Conversion primary to conservative variables
//! @param[inout] rho density array 3D
//! @param[inout] rhou momentum array 3D
//! @param[inout] E energy array 3D
//! @param[in] u speed array 3D
//! @param[in] P pressure array 3D
//! @param[in] gamma
void ConvPrimConsArray(
    Kokkos::View<double***> const rho,
    Kokkos::View<double***> const rhou,
    Kokkos::View<double***> const E,
    Kokkos::View<double***> const u,
    Kokkos::View<double***> const P,
    double const gamma);
    
//! Conversion conservative to primary variables
//! @param[inout] rho density array 3D
//! @param[inout] rhou momentum array 3D
//! @param[inout] E energy array 3D
//! @param[in] u speed array 3D
//! @param[in] P pressure array 3D
//! @param[in] gamma
void ConvConsPrimArray(
    Kokkos::View<double***> const rho,
    Kokkos::View<double***> const rhou,
    Kokkos::View<double***> const E,
    Kokkos::View<double***> const u,
    Kokkos::View<double***> const P,
    double const gamma);