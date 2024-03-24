//!
//! @file riemann_solver.hpp
//!

#pragma once

#include <Kokkos_Core.hpp>
#include <eos.hpp>

#include "euler_equations.hpp"

namespace novapp
{

class HLL
{
public:
    //! HLL solver
    //! @param[in]
    //! @return intercell EulerFlux
    KOKKOS_FORCEINLINE_FUNCTION
    EulerFlux operator()(
            EulerCons const& consL,
            EulerCons const& consR,
            int locdim,
            EOS const& eos) const noexcept
    {
        KOKKOS_ASSERT(locdim >= 0)
        KOKKOS_ASSERT(locdim < ndim)
        EulerPrim const primL = to_prim(consL, eos);
        EulerPrim const primR = to_prim(consR, eos);

        double const cL = eos.compute_speed_of_sound(primL.rho, primL.P);
        double const cR = eos.compute_speed_of_sound(primR.rho, primR.P);

        double const wsL = Kokkos::fmin(primL.u[locdim] - cL, primR.u[locdim] - cR);
        double const wsR = Kokkos::fmax(primL.u[locdim] + cL, primR.u[locdim] + cR);

        double const neg_wsL = Kokkos::fmin(wsL, 0.);
        double const pos_wsR = Kokkos::fmax(wsR, 0.);

        EulerFlux const fluxL = compute_flux(primL, locdim, eos);
        EulerFlux const fluxR = compute_flux(primR, locdim, eos);

        EulerFlux flux;
        flux.rho = FluxHLL(consL.rho, consR.rho, fluxL.rho, fluxR.rho, neg_wsL, pos_wsR);
        for (int idim = 0; idim < ndim; ++idim)
        {
            flux.rhou[idim] = FluxHLL(
                consL.rhou[idim],
                consR.rhou[idim],
                fluxL.rhou[idim],
                fluxR.rhou[idim],
                neg_wsL,
                pos_wsR);
        }
        flux.E = FluxHLL(consL.E, consR.E, fluxL.E, fluxR.E, neg_wsL, pos_wsR);
        return flux;
    }

    KOKKOS_FORCEINLINE_FUNCTION
    static double FluxHLL(
            double const UL,
            double const UR,
            double const FL,
            double const FR,
            double const wsL,
            double const wsR) noexcept
    {
        return (wsR * FL - wsL * FR + wsL * wsR * (UR - UL)) / (wsR - wsL);
    }
};

class HLLC
{
public:
    KOKKOS_FORCEINLINE_FUNCTION
    EulerFlux operator()(
            EulerCons const& consL,
            EulerCons const& consR,
            int locdim,
            EOS const& eos) const noexcept
    {
        KOKKOS_ASSERT(locdim >= 0)
        KOKKOS_ASSERT(locdim < ndim)
        EulerPrim const primL = to_prim(consL, eos);
        EulerPrim const primR = to_prim(consR, eos);

        double const cL = eos.compute_speed_of_sound(primL.rho, primL.P);
        double const cR = eos.compute_speed_of_sound(primR.rho, primR.P);

        double const wsL = Kokkos::fmin(primL.u[locdim] - cL, primR.u[locdim] - cR);
        double const wsR = Kokkos::fmax(primL.u[locdim] + cL, primR.u[locdim] + cR);

        EulerFlux const fluxL = compute_flux(primL, locdim, eos);
        EulerFlux const fluxR = compute_flux(primR, locdim, eos);

        double const rcL = primL.rho * (wsL - primL.u[locdim]);
        double const rcR = primR.rho * (wsR - primR.u[locdim]);

        double const ustar = (primR.P - primL.P + rcL * primL.u[locdim] - rcR * primR.u[locdim]) 
                            / (rcL - rcR);
        double const pstar = 1. / 2 * (primL.P + primR.P + rcL * (ustar - primL.u[locdim])
                            + rcR * (ustar - primR.u[locdim]));

        double S = 0;
        EulerCons cons_state;
        EulerPrim prim_state;
        EulerFlux flux_state;

        if (ustar > 0)
        {
            S = wsL;
            cons_state = consL;
            prim_state = primL;
            flux_state = fluxL;
        }
        else
        {
            S = wsR;
            cons_state = consR;
            prim_state = primR;
            flux_state = fluxR;
        }

        double un = prim_state.u[locdim];
        EulerFlux flux;

        if (wsL * wsR > 0)
        {
            flux.rho = flux_state.rho;
            for (int idim = 0; idim < ndim; ++idim)
            {
                flux.rhou[idim] = flux_state.rhou[idim];
            }
            flux.E = flux_state.E;
        }
        else
        {
            double rho_star = cons_state.rho * (S - un) / (S - ustar);
            flux.rho = FluxHLLC(flux_state.rho, cons_state.rho, rho_star, S);

            for (int idim = 0; idim < ndim; ++idim)
            {
                flux.rhou[idim] = rho_star * ustar * prim_state.u[idim];
            }
            flux.rhou[locdim] = rho_star * ustar * ustar + pstar;

            double E_star = (S - un) / (S - ustar) * cons_state.E
                            + (ustar - un) / (S - ustar)
                            * (ustar * cons_state.rho * (S - un) + prim_state.P);
            flux.E = FluxHLLC(flux_state.E, cons_state.E, E_star, S);
        }
        return flux;
    }

    KOKKOS_FORCEINLINE_FUNCTION
    static double FluxHLLC(
            double const F,
            double const U,
            double const Ustar,
            double const ws) noexcept
    {
        return F + ws * (Ustar - U);
    }
};

class Splitting
{
public:
    KOKKOS_FORCEINLINE_FUNCTION
    EulerFlux operator()(
            EulerCons const& consL,
            EulerCons const& consR,
            int locdim,
            EOS const& eos) const noexcept
    {
        KOKKOS_ASSERT(locdim >= 0)
        KOKKOS_ASSERT(locdim < ndim)
        EulerPrim const primL = to_prim(consL, eos);
        EulerPrim const primR = to_prim(consR, eos);

        double const cL = eos.compute_speed_of_sound(primL.rho, primL.P);
        double const cR = eos.compute_speed_of_sound(primR.rho, primR.P);

        // Low Mach correction
        double a = 1.1 * Kokkos::fmax(primL.rho * cL, primR.rho * cR);
        double ustar = (primL.u[locdim] + primR.u[locdim]) / 2 - 1 / (2 * a) * (primR.P - primL.P);
        double Ma = Kokkos::fabs(ustar) / Kokkos::fmin(cL, cR);
        double theta = Kokkos::fmin(1, Ma);
        double Pstar = (primL.P + primR.P) / 2 - (theta * a) / 2 * (primR.u[locdim] - primL.u[locdim]);

        EulerCons cons_state;
        if (ustar > 0)
        {
            cons_state = consL;
        }
        else
        {
            cons_state = consR;
        }

        EulerFlux flux;
        flux.rho = ustar * cons_state.rho;
        for (int idim = 0; idim < ndim; ++idim)
        {
            flux.rhou[idim] = ustar * cons_state.rhou[idim];
        }
        flux.rhou[locdim] += Pstar;
        flux.E = ustar * cons_state.E + Pstar * ustar;
        return flux;
    }
};

} // namespace novapp
