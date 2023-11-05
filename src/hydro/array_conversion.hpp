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

//! Conversion from primitive to conservative variables
//! @param[in] eos equation of state
//! @param[inout] rho density array 3D
//! @param[in] u velocity array 3D
//! @param[in] P pressure array 3D
//! @param[inout] rhou momentum array 3D
//! @param[inout] E total energy array 3D
void conv_prim_to_cons(
    Range const& range,
    EOS const& eos,
    Kokkos::View<const double***, Kokkos::LayoutStride> rho,
    Kokkos::View<const double****, Kokkos::LayoutStride> u,
    Kokkos::View<const double***, Kokkos::LayoutStride> P,
    Kokkos::View<double****, Kokkos::LayoutStride> rhou,
    Kokkos::View<double***, Kokkos::LayoutStride> E);


//! Conversion from conservative to primitive variables
//! @param[in] eos equation of state
//! @param[in] rho density array 3D
//! @param[in] rhou momentum array 3D
//! @param[in] E total energy array 3D
//! @param[inout] u velocity array 3D
//! @param[inout] P pressure array 3D
void conv_cons_to_prim(
    Range const& range,
    EOS const& eos,
    KV_cdouble_3d rho,
    KV_cdouble_4d rhou,
    KV_cdouble_3d E,
    KV_double_4d u,
    KV_double_3d P);

} // namespace novapp
