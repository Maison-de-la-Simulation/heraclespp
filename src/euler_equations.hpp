//!
//! @file euler_equations.hpp
//!

#pragma once

#include <Kokkos_Core.hpp>
#include <PerfectGas.hpp>
#include <RadGas.hpp>

#include "kronecker.hpp"
#include "ndim.hpp"
#include "eos.hpp"

namespace novapp
{

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

KOKKOS_INLINE_FUNCTION
double compute_ek(EulerCons const& cons) noexcept
{
    double norm_rhou = 0;
    for (int idim = 0; idim < ndim; ++idim)
    {
        norm_rhou += cons.rhou[idim] * cons.rhou[idim];
    }
    return 0.5 * norm_rhou / cons.rho;
}

KOKKOS_INLINE_FUNCTION
double compute_ek(EulerPrim const& prim) noexcept
{
    double norm_u = 0;
    for (int idim = 0; idim < ndim; ++idim)
    {
        norm_u += prim.u[idim] * prim.u[idim];
    }
    return 0.5 * prim.rho * norm_u;
}

//! Flux formula
//! @param[in] prim Primitive state
//! @param[in] eos Equation of state
//! @return flux
KOKKOS_INLINE_FUNCTION
EulerFlux compute_flux(
        EulerPrim const& prim,
        int locdim,
        EOS const& eos) noexcept
{
    EulerFlux flux;
    double const volumic_total_energy
            = compute_ek(prim) + eos.compute_evol_from_P(prim.rho, prim.P);
    flux.rho = prim.rho * prim.u[locdim];
    for (int idim = 0; idim < ndim; ++idim)
    {
        flux.rhou[idim] = prim.rho * prim.u[locdim] * prim.u[idim];
    }
    flux.rhou[locdim] += prim.P;
    flux.E = prim.u[locdim] * (volumic_total_energy + prim.P);
    return flux;
}

//! Flux formula
//! @param[in] cons Conservative state
//! @param[in] eos Equation of state
//! @return flux
KOKKOS_INLINE_FUNCTION
EulerFlux compute_flux(
        EulerCons const& cons,
        int locdim,
        EOS const& eos) noexcept
{
    EulerFlux flux;
    double const evol = cons.E - compute_ek(cons);
    double const P = eos.compute_P_from_evol(cons.rho, evol);
    double const u = cons.rhou[locdim] / cons.rho;
    flux.rho = u * cons.rho;
    for (int idim = 0; idim < ndim; ++idim)
    {
        flux.rhou[idim] = cons.rhou[locdim] * cons.rhou[idim] / cons.rho;
    }
    flux.rhou[locdim] += P;
    flux.E = u * (cons.E + P);
    return flux;
}

KOKKOS_INLINE_FUNCTION
EulerPrim to_prim(
        EulerCons const& cons, 
        EOS const& eos) noexcept
{
    EulerPrim prim;
    double const evol = cons.E - compute_ek(cons);
    prim.rho = cons.rho;
    for (int idim = 0; idim < ndim; ++idim)
    {
        prim.u[idim] = cons.rhou[idim] / cons.rho;
    }
    prim.P = eos.compute_P_from_evol(cons.rho, evol);
    return prim;
}

KOKKOS_INLINE_FUNCTION
EulerCons to_cons(EulerPrim const& prim, EOS const& eos) noexcept
{
    EulerCons cons;
    cons.rho = prim.rho;
    for (int idim = 0; idim < ndim; ++idim)
    {
        cons.rhou[idim] = prim.rho * prim.u[idim];
    }
    cons.E = eos.compute_evol_from_P(prim.rho, prim.P) 
                + compute_ek(prim);
    return cons;
}

} // namespace novapp
