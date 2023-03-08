/**
 * @file array_conversion.hpp
 * General fonctions
 */
#pragma once

#include <Kokkos_Core.hpp>

namespace thermodynamics
{

class PerfectGas;

} // namespace thermodynamics

class Range;

//! Conversion primary to conservative variables
//! @param[inout] rhou momentum array 3D
//! @param[inout] E energy array 3D
//! @param[inout] rho density array 3D
//! @param[in] u speed array 3D
//! @param[in] P pressure array 3D
//! @param[in] eos equation of state
void ConvPrimtoConsArray(
    Range const& range,
    Kokkos::View<double****, Kokkos::LayoutStride> const rhou,
    Kokkos::View<double***, Kokkos::LayoutStride> const E,
    Kokkos::View<const double***, Kokkos::LayoutStride> const rho,
    Kokkos::View<const double****, Kokkos::LayoutStride> const u,
    Kokkos::View<const double***, Kokkos::LayoutStride> const P,
    thermodynamics::PerfectGas const& eos);
    
//! Conversion conservative to primary variables
//! @param[inout] rho density array 3D
//! @param[inout] rhou momentum array 3D
//! @param[inout] E energy array 3D
//! @param[in] u speed array 3D
//! @param[in] P pressure array 3D
//! @param[in] eos equation of state
void ConvConstoPrimArray(
    Range const& range,
    Kokkos::View<double****, Kokkos::LayoutStride> const u,
    Kokkos::View<double***, Kokkos::LayoutStride> const P,
    Kokkos::View<const double***, Kokkos::LayoutStride> const rho,
    Kokkos::View<const double****, Kokkos::LayoutStride> const rhou,
    Kokkos::View<const double***, Kokkos::LayoutStride> const E,
    thermodynamics::PerfectGas const& eos);
