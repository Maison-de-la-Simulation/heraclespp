// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file array_conversion.hpp
//! General functions
//!

#pragma once

#include <cassert>

#include <Kokkos_Core.hpp>
#include <kokkos_shortcut.hpp>
#include <ndim.hpp>
#include <range.hpp>

#include "concepts.hpp"
#include "euler_equations.hpp"

namespace novapp {

//! Conversion from primitive to conservative variables
//! @param[in] range output iteration range
//! @param[in] eos equation of state
//! @param[in] rho density array 3D
//! @param[in] u velocity array 3D
//! @param[in] P pressure array 3D
//! @param[inout] rhou momentum array 3D
//! @param[inout] E total energy array 3D
template <concepts::EulerEoS EoS>
void conv_prim_to_cons(
        Range const& range,
        EoS const& eos,
        KV_cdouble_3d const& rho,
        Kokkos::Array<KV_cdouble_3d, ndim> const& u,
        KV_cdouble_3d const& P,
        Kokkos::Array<KV_double_3d, ndim> const& rhou,
        KV_double_3d const& E)
{
    assert(equal_extents({0, 1, 2}, rho, rhou[0], E, u[0], P));
    assert(equal_extents({0, 1, 2}, rhou));
    assert(equal_extents({0, 1, 2}, u));

    Kokkos::parallel_for(
            "conv_prim_to_cons_array",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k) {
                EulerPrim prim;
                prim.rho = rho(i, j, k);
                for (int idim = 0; idim < ndim; ++idim) {
                    prim.u[idim] = u[idim](i, j, k);
                }
                prim.P = P(i, j, k);

                EulerCons const cons = to_cons(prim, eos);
                for (int idim = 0; idim < ndim; ++idim) {
                    rhou[idim](i, j, k) = cons.rhou[idim];
                }
                E(i, j, k) = cons.E;
            });
}

//! Conversion from primitive to conservative variables
//! @param[in] range output iteration range
//! @param[in] eos equation of state
//! @param[in] rho density array 3D
//! @param[in] u velocity array 3D
//! @param[in] P pressure array 3D
//! @param[inout] rhou momentum array 3D
//! @param[inout] E total energy array 3D
template <concepts::EulerEoS EoS>
void conv_prim_to_cons(
        Range const& range,
        EoS const& eos,
        KV_cdouble_3d const& rho,
        KV_cdouble_4d const& u,
        KV_cdouble_3d const& P,
        KV_double_4d const& rhou,
        KV_double_3d const& E)
{
    assert(equal_extents({0, 1, 2}, rho, rhou, E, u, P));
    assert(equal_extents(3, rhou, u));
    assert(u.extent_int(3) == ndim);

    Kokkos::Array<KV_cdouble_3d, ndim> u_array;
    for (int iv = 0; iv < ndim; ++iv) {
        u_array[iv] = Kokkos::subview(u, ALL, ALL, ALL, iv);
    }
    Kokkos::Array<KV_double_3d, ndim> rhou_array;
    for (int iv = 0; iv < ndim; ++iv) {
        rhou_array[iv] = Kokkos::subview(rhou, ALL, ALL, ALL, iv);
    }

    conv_prim_to_cons(range, eos, rho, u_array, P, rhou_array, E);
}

//! Conversion from conservative to primitive variables
//! @param[in] range output iteration range
//! @param[in] eos equation of state
//! @param[in] rho density array 3D
//! @param[in] rhou momentum array 3D
//! @param[in] E total energy array 3D
//! @param[inout] u velocity array 3D
//! @param[inout] P pressure array 3D
template <concepts::EulerEoS EoS>
void conv_cons_to_prim(
        Range const& range,
        EoS const& eos,
        KV_cdouble_3d const& rho,
        Kokkos::Array<KV_cdouble_3d, ndim> const& rhou,
        KV_cdouble_3d const& E,
        Kokkos::Array<KV_double_3d, ndim> const& u,
        KV_double_3d const& P)
{
    assert(equal_extents({0, 1, 2}, rho, rhou[0], E, u[0], P));
    assert(equal_extents({0, 1, 2}, rhou));
    assert(equal_extents({0, 1, 2}, u));

    Kokkos::parallel_for(
            "conv_cons_to_prim_array",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k) {
                EulerCons cons;
                cons.rho = rho(i, j, k);
                for (int idim = 0; idim < ndim; ++idim) {
                    cons.rhou[idim] = rhou[idim](i, j, k);
                }
                cons.E = E(i, j, k);

                EulerPrim const prim = to_prim(cons, eos);
                for (int idim = 0; idim < ndim; ++idim) {
                    u[idim](i, j, k) = prim.u[idim];
                }
                P(i, j, k) = prim.P;
            });
}

//! Conversion from conservative to primitive variables
//! @param[in] range output iteration range
//! @param[in] eos equation of state
//! @param[in] rho density array 3D
//! @param[in] rhou momentum array 3D
//! @param[in] E total energy array 3D
//! @param[inout] u velocity array 3D
//! @param[inout] P pressure array 3D
template <concepts::EulerEoS EoS>
void conv_cons_to_prim(
        Range const& range,
        EoS const& eos,
        KV_cdouble_3d const& rho,
        KV_cdouble_4d const& rhou,
        KV_cdouble_3d const& E,
        KV_double_4d const& u,
        KV_double_3d const& P)
{
    assert(equal_extents({0, 1, 2}, rho, rhou, E, u, P));
    assert(equal_extents(3, rhou, u));
    assert(u.extent_int(3) == ndim);

    Kokkos::Array<KV_cdouble_3d, ndim> rhou_array;
    for (int iv = 0; iv < ndim; ++iv) {
        rhou_array[iv] = Kokkos::subview(rhou, ALL, ALL, ALL, iv);
    }
    Kokkos::Array<KV_double_3d, ndim> u_array;
    for (int iv = 0; iv < ndim; ++iv) {
        u_array[iv] = Kokkos::subview(u, ALL, ALL, ALL, iv);
    }

    conv_cons_to_prim(range, eos, rho, rhou_array, E, u_array, P);
}

} // namespace novapp
