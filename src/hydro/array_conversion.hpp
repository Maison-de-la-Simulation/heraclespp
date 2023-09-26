//!
//! @file array_conversion.hpp
//! General fonctions
//!

#pragma once

#include <eos.hpp>
#include <kokkos_shortcut.hpp>

namespace novapp
{

namespace thermodynamics
{

class PerfectGas;
class RadGas;

} // namespace thermodynamics

class Range;

//! Conversion primary to conservative variables
//! @param[inout] rhou momentum array 3D
//! @param[inout] E energy array 3D
//! @param[inout] rho density array 3D
//! @param[in] u speed array 3D
//! @param[in] P pressure array 3D
//! @param[in] eos equation of state
void conv_prim_to_cons(
    Range const& range,
    Kokkos::View<double****, Kokkos::LayoutStride> rhou,
    Kokkos::View<double***, Kokkos::LayoutStride> E,
    Kokkos::View<const double***, Kokkos::LayoutStride> rho,
    Kokkos::View<const double****, Kokkos::LayoutStride> u,
    Kokkos::View<const double***, Kokkos::LayoutStride> P,
    EOS const& eos);


//! Conversion conservative to primary variables
//! @param[inout] rho density array 3D
//! @param[inout] rhou momentum array 3D
//! @param[inout] E energy array 3D
//! @param[in] u speed array 3D
//! @param[in] P pressure array 3D
//! @param[in] eos equation of state
void conv_cons_to_prim(
    Range const& range,
    KV_double_4d u,
    KV_double_3d P,
    KV_cdouble_3d rho,
    KV_cdouble_4d rhou,
    KV_cdouble_3d E,
    EOS const& eos);

} // namespace novapp
