//!
//! @file source_terms.hpp
//!

#pragma once

#include <Kokkos_Macros.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include "euler_equations.hpp"

namespace novapp
{

// Pressure term (e_{r}): 2 * P_{rr} / r
// Pressure term (e_{th}): cot(th) * P_{th th} / r
KOKKOS_FORCEINLINE_FUNCTION
double source_grad_P(
    double const dtodv,
    EulerPrim const& primL,
    EulerPrim const& primR,
    double const dsL,
    double const dsR)
{
    return dtodv * (primL.P + primR.P) / 2 * (dsR - dsL);
}

// Velocity term (e_{r}): rho * u_{th} * u_{th} / r
// Velocity term (e_{r}): rho * u_{phi} * u_{phi} / r
KOKKOS_FORCEINLINE_FUNCTION
double source_grad_u_r(
    int const locdim,
    double const dtodv,
    EulerPrim const& primL,
    EulerPrim const& primR,
    double const dsL,
    double const dsR)
{
    return dtodv * (primL.rho * primL.u[locdim] * primL.u[locdim] + primR.rho * primR.u[locdim] * primR.u[locdim] ) / 2 * (dsR - dsL) / 2;
}

// Velocity term (e_{th}): rho * u_{th} * u_{r} / r
// Velocity term (e_{phi}): rho * u_{phi} * u_{r} / r
KOKKOS_FORCEINLINE_FUNCTION
double source_grad_u_idir_r(
    int const locdim,
    double const dtodv,
    double const xL,
    double const xR,
    EulerPrim const& primL,
    EulerPrim const& primR,
    double const dsL,
    double const dsR)
{
    return dtodv * (xR - xL) / (xR + xL)
        * (primR.rho * primR.u[locdim] * primR.u[0] * dsR + primL.rho * primL.u[locdim] * primL.u[0] * dsL);
}

// Velocity term (e_{th}): cot(th) * rho * u_{phi} * u_{phi} / r
KOKKOS_FORCEINLINE_FUNCTION
double source_grad_u_th(
    double const dtodv,
    double const yL,
    double const yR,
    EulerPrim const& primL,
    EulerPrim const& primR,
    double const dsL,
    double const dsR)
{
    double const y_mid = (yL + yR) / 2;
    return dtodv * (primL.rho * primL.u[2] * primL.u[2] + primR.rho * primR.u[2] * primR.u[2]) / 2
        * (Kokkos::cos(y_mid) / Kokkos::sin(y_mid))
        * (dsL - dsR) / 2;
}

// Velocity term (e_{phi}): cot(th) * rho * u_{phi} * u_{th} / r
KOKKOS_FORCEINLINE_FUNCTION
double source_grad_u_phi(
    double const dtodv,
    double const yL,
    double const yR,
    EulerPrim const& primL,
    EulerPrim const& primR,
    double const dsL,
    double const dsR)
{
    double const sm = Kokkos::sin(yL);
    double const sp = Kokkos::sin(yR);
    return dtodv * (sp - sm) / (sp + sm) * (primR.rho * primR.u[2] * primR.u[1] * dsL
        + primL.rho * primL.u[2] * primL.u[1] * dsR);
}

} // namespace novapp
