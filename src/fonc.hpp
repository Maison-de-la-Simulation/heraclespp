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
//! @param[in] nx number of cells
//! @param[in] gamma
void ConvPrimCons(
        Kokkos::View<double*> const rho,
        Kokkos::View<double*> const rhou,
        Kokkos::View<double*> const E,
        Kokkos::View<double*> const u,
        Kokkos::View<double*> const P,
        int const nx,
        double const gamma);

//! Conversion conservative to primary variables
//! @param[inout] rho density array
//! @param[inout] rhou momentum array
//! @param[inout] E energy array
//! @param[in] u speed array
//! @param[in] P pressure array
//! @param[in] nx number of cells
//! @param[in] gamma
void ConvConsPrim(
        Kokkos::View<double*> const rho,
        Kokkos::View<double*> const rhou,
        Kokkos::View<double*> const E,
        Kokkos::View<double*> const u,
        Kokkos::View<double*> const P,
        int const nx,
        double const gamma);

//! Van Leer slope formule
//! @param[in] DiffL = U_{i+1} - U_{i}
//! @param[in] DiffR = U_{i} - U_{i-1}
//! @return slope
double VanLeer(
        double DiffL,
        double DiffR);

//! Minod slope formule
//! @param[in] DiffL = U_{i+1} - U_{i}
//! @param[in] DiffR = U_{i} - U_{i-1}
//! @return slope
double Minmod(
        double DiffL,
        double DiffR);

//! Van Albada slope formule
//! @param[in] DiffL = U_{i+1} - U_{i}
//! @param[in] DiffR = U_{i} - U_{i-1}
//! @return slope
double VanAlbada(
        double DiffL,
        double DiffR);

//! Flux for conservative variable rho
//! @param[in] rho density array
//! @param[in] u speed
//! @return flux
double FluxRho(
        double const rho,
        double const u);

//! Flux for conservative variable rho * u
//! @param[in] rho density array
//! @param[in] u speed
//! @param[in] P pressure
//! @return flux
double FluxRhou(
        double const rho,
        double const u,
        double const P);

//! Flux for conservative variable E
//! @param[in] rho density array
//! @param[in] u speed
//! @param[in] P pressure
//! @param[in] gamma
//! @return flux
double FluxE(
        double const rho,
        double const u,
        double const P,
        double const gamma);
