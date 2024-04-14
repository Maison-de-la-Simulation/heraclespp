//!
//! @file array_conversion.hpp
//! General fonctions
//!

#pragma once

#include <kokkos_shortcut.hpp>
#include <range.hpp>

#include "euler_equations.hpp"

namespace novapp
{

//! Conversion from primitive to conservative variables
//! @param[in] range output iteration range
//! @param[in] eos equation of state
//! @param[in] rho density array 3D
//! @param[in] u velocity array 3D
//! @param[in] P pressure array 3D
//! @param[inout] rhou momentum array 3D
//! @param[inout] E total energy array 3D
template<class EoS>
void conv_prim_to_cons(
    Range const& range,
    EoS const& eos,
    Kokkos::View<const double***, Kokkos::LayoutStride> const rho,
    Kokkos::View<const double****, Kokkos::LayoutStride> const u,
    Kokkos::View<const double***, Kokkos::LayoutStride> const P,
    Kokkos::View<double****, Kokkos::LayoutStride> const rhou,
    Kokkos::View<double***, Kokkos::LayoutStride> const E)
{
    Kokkos::parallel_for(
        "conv_prim_to_cons_array",
        cell_mdrange(range),
        KOKKOS_LAMBDA(int i, int j, int k)
        {
            EulerPrim prim;
            prim.rho = rho(i, j, k);
            for (int idim = 0; idim < ndim; ++idim)
            {
                prim.u[idim] = u(i, j, k, idim);
            }
            prim.P = P(i, j, k);

            EulerCons const cons = to_cons(prim, eos);
            for (int idim = 0; idim < ndim; ++idim)
            {
                rhou(i, j, k, idim) = cons.rhou[idim];
            }
            E(i, j, k) = cons.E;
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
template<class EoS>
void conv_cons_to_prim(
    Range const& range,
    EoS const& eos,
    KV_cdouble_3d const rho,
    KV_cdouble_4d const rhou,
    KV_cdouble_3d const E,
    KV_double_4d const u,
    KV_double_3d const P)
{
    Kokkos::parallel_for(
        "conv_cons_to_prim_array",
        cell_mdrange(range),
        KOKKOS_LAMBDA(int i, int j, int k)
        {
            EulerCons cons;
            cons.rho = rho(i, j, k);
            for (int idim = 0; idim < ndim; ++idim)
            {
                cons.rhou[idim] = rhou(i, j, k, idim);
            }
            cons.E = E(i, j, k);

            EulerPrim const prim = to_prim(cons, eos);
            for (int idim = 0; idim < ndim; ++idim)
            {
                u(i, j, k, idim) = prim.u[idim];
            }
            P(i, j, k) = prim.P;
        });
}

} // namespace novapp
