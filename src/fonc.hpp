/**
 * @file fonc.hpp
 * General fonctions
 */
#pragma once

#include <Kokkos_Core.hpp>

//! Conversion primary to conservative variables
//! @param[inout] rho density array
//! @param[inout] rhou momentum array
//! @param[inout] E energy array
//! @param[in] u speed array
//! @param[in] P pressure array
//! @param[in] gamma
void ConvPrimCons(
        Kokkos::View<double*> const rho,
        Kokkos::View<double*> const rhou,
        Kokkos::View<double*> const E,
        Kokkos::View<double*> const u,
        Kokkos::View<double*> const P,
        double const gamma);

//! Conversion conservative to primary variables
//! @param[inout] rho density array
//! @param[inout] rhou momentum array
//! @param[inout] E energy array
//! @param[in] u speed array
//! @param[in] P pressure array
//! @param[in] gamma
void ConvConsPrim(
        Kokkos::View<double*> const rho,
        Kokkos::View<double*> const rhou,
        Kokkos::View<double*> const E,
        Kokkos::View<double*> const u,
        Kokkos::View<double*> const P,
        double const gamma);
