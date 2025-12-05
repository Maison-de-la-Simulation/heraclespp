// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file euler_equations.hpp
//!

#pragma once

#include <Kokkos_Array.hpp>
#include <Kokkos_Assert.hpp>
#include <Kokkos_Macros.hpp>
#include <ndim.hpp>

#include "concepts.hpp"

namespace hclpp {

struct EulerCons
{
    double rho;
    Kokkos::Array<double, ndim> rhou;
    double E;
};

struct EulerFlux
{
    double rho;
    Kokkos::Array<double, ndim> rhou;
    double E;
};

struct EulerPrim
{
    double rho;
    Kokkos::Array<double, ndim> u;
    double P;
};

KOKKOS_FORCEINLINE_FUNCTION
double compute_ek(EulerCons const& cons) noexcept
{
    double norm_rhou = 0;
    for (int idim = 0; idim < ndim; ++idim) {
        norm_rhou += cons.rhou[idim] * cons.rhou[idim];
    }
    return norm_rhou / cons.rho / 2;
}

KOKKOS_FORCEINLINE_FUNCTION
double compute_ek(EulerPrim const& prim) noexcept
{
    double norm_u = 0;
    for (int idim = 0; idim < ndim; ++idim) {
        norm_u += prim.u[idim] * prim.u[idim];
    }
    return prim.rho * norm_u / 2;
}

KOKKOS_FORCEINLINE_FUNCTION
double compute_evol(EulerCons const& cons) noexcept
{
    return cons.E - compute_ek(cons);
}

//! Flux formula
//! @param[in] prim Primitive state
//! @param[in] locdim index of the direction X, Y or Z
//! @param[in] eos Equation of state
//! @return flux
template <concepts::EulerEoS EoS>
KOKKOS_FORCEINLINE_FUNCTION EulerFlux compute_flux(EulerPrim const& prim, int locdim, EoS const& eos) noexcept
{
    KOKKOS_ASSERT(locdim >= 0)
    KOKKOS_ASSERT(locdim < ndim)
    EulerFlux flux;
    double const E = compute_ek(prim) + eos.compute_evol_from_pres(prim.rho, prim.P);
    flux.rho = prim.rho * prim.u[locdim];
    for (int idim = 0; idim < ndim; ++idim) {
        flux.rhou[idim] = prim.rho * prim.u[locdim] * prim.u[idim];
    }
    flux.rhou[locdim] += prim.P;
    flux.E = prim.u[locdim] * (E + prim.P);
    return flux;
}

//! Flux formula
//! @param[in] cons Conservative state
//! @param[in] locdim index of the direction X, Y or Z
//! @param[in] eos Equation of state
//! @return flux
template <concepts::EulerEoS EoS>
KOKKOS_FORCEINLINE_FUNCTION EulerFlux compute_flux(EulerCons const& cons, int locdim, EoS const& eos) noexcept
{
    KOKKOS_ASSERT(locdim >= 0)
    KOKKOS_ASSERT(locdim < ndim)
    EulerFlux flux;
    double const evol = compute_evol(cons);
    double const P = eos.compute_pres_from_evol(cons.rho, evol);
    double const u = cons.rhou[locdim] / cons.rho;
    flux.rho = u * cons.rho;
    for (int idim = 0; idim < ndim; ++idim) {
        flux.rhou[idim] = u * cons.rhou[idim];
    }
    flux.rhou[locdim] += P;
    flux.E = u * (cons.E + P);
    return flux;
}

template <concepts::EulerEoS EoS>
KOKKOS_FORCEINLINE_FUNCTION EulerPrim to_prim(EulerCons const& cons, EoS const& eos) noexcept
{
    EulerPrim prim;
    double const evol = compute_evol(cons);
    prim.rho = cons.rho;
    for (int idim = 0; idim < ndim; ++idim) {
        prim.u[idim] = cons.rhou[idim] / cons.rho;
    }
    prim.P = eos.compute_pres_from_evol(cons.rho, evol);
    return prim;
}

template <concepts::EulerEoS EoS>
KOKKOS_FORCEINLINE_FUNCTION EulerCons to_cons(EulerPrim const& prim, EoS const& eos) noexcept
{
    EulerCons cons;
    cons.rho = prim.rho;
    for (int idim = 0; idim < ndim; ++idim) {
        cons.rhou[idim] = prim.rho * prim.u[idim];
    }
    cons.E = eos.compute_evol_from_pres(prim.rho, prim.P) + compute_ek(prim);
    return cons;
}

} // namespace hclpp
