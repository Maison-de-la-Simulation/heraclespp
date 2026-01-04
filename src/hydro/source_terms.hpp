// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file source_terms.hpp
//!

#pragma once

#include <Kokkos_Macros.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

namespace hclpp {

// Pressure term (e_{r}): 2 * P_{rr} / r
// Pressure term (e_{th}): cot(th) * P_{th th} / r
KOKKOS_FORCEINLINE_FUNCTION
auto source_grad_pres(double dtodv, double PL, double PR, double dS, double dS_p) -> double
{
    return dtodv * (PL + PR) / 2 * (dS_p - dS);
}

// Velocity term (e_{r}): rho * u_{th} * u_{th} / r
// Velocity term (e_{r}): rho * u_{phi} * u_{phi} / r
KOKKOS_FORCEINLINE_FUNCTION
auto source_grad_u_r(double dtodv, double rhoL, double rhoR, double uL, double uR, double dS, double dS_p) -> double
{
    return dtodv * (rhoL * uL * uL + rhoR * uR * uR) / 2 * (dS_p - dS) / 2;
}

// Velocity term (e_{th}): rho * u_{th} * u_{r} / r
// Velocity term (e_{phi}): rho * u_{phi} * u_{r} / r
KOKKOS_FORCEINLINE_FUNCTION
auto source_grad_u_idir_r(
        double dtodv,
        double x0,
        double x0_p,
        double rhoL,
        double rhoR,
        double uL_r,
        double uR_r,
        double uL_other,
        double uR_other,
        double dS,
        double dS_p) -> double
{
    return dtodv * (x0_p - x0) / (x0_p + x0) * (rhoR * uR_other * uR_r * dS_p + rhoL * uL_other * uL_r * dS);
}

// Velocity term (e_{th}): cot(th) * rho * u_{phi} * u_{phi} / r
KOKKOS_FORCEINLINE_FUNCTION
auto source_grad_u_th(double dtodv, double x1, double x1_p, double rhoL, double rhoR, double uL_phi, double uR_phi, double dS, double dS_p) -> double
{
    return dtodv * (rhoL * uL_phi * uL_phi + rhoR * uR_phi * uR_phi) / 2 * (Kokkos::cos((x1 + x1_p) / 2) / Kokkos::sin((x1 + x1_p) / 2)) * (dS_p - dS)
           / 2;
}

// Velocity term (e_{phi}): cot(th) * rho * u_{phi} * u_{th} / r
KOKKOS_FORCEINLINE_FUNCTION
auto source_grad_u_phi(
        double dtodv,
        double x1,
        double x1_p,
        double rhoL,
        double rhoR,
        double uL_th,
        double uR_th,
        double uL_phi,
        double uR_phi,
        double dS,
        double dS_p) -> double
{
    double const sm = Kokkos::sin(x1);
    double const sp = Kokkos::sin(x1_p);
    return dtodv * (sp - sm) / (sp + sm) * (rhoR * uR_phi * uR_th * dS_p + rhoL * uL_phi * uL_th * dS);
}

} // namespace hclpp
