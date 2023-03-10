/**
 * @file array_conversion.hpp
 * General fonctions
 */
#pragma once

#include "Kokkos_shortcut.hpp"

namespace novapp
{

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
    KV_double_4d const u,
    KV_double_3d const P,
    KV_cdouble_3d const rho,
    KV_cdouble_4d const rhou,
    KV_cdouble_3d const E,
    thermodynamics::PerfectGas const& eos);

} // namespace novapp
