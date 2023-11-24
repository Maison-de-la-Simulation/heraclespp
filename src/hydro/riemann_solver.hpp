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
        assert(locdim >= 0);
        assert(locdim < ndim);
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
        assert(locdim >= 0);
        assert(locdim < ndim);
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

        double const rcL = primL.rho * (wsL - primL.u[locdim]);
        double const rcR = primR.rho * (wsR - primR.u[locdim]);

        // Compute star states
        double const ustar = (primR.P - primL.P + rcL * primL.u[locdim] - rcR * primR.u[locdim]) 
                                / (rcL - rcR);

        double rho_star = 0;
        double rhou_star = 0;
        double E_star = 0;
        double S = 0;
        EulerCons cons_state;
        EulerFlux flux_state;

        if (ustar > 0)
        {
            S = wsL;
            cons_state = consL;
            flux_state = fluxL;
            rho_star = consL.rho * (wsL - primL.u[locdim]) / (wsL - ustar);
            rhou_star = rho_star * ustar;
            E_star = rho_star * (consL.E / consL.rho + (ustar - primL.u[locdim]) * (ustar + primL.P / rcL));
        }
        else
        {
            S = wsR;
            cons_state = consR;
            flux_state = fluxR;
            rho_star = consR.rho * (wsR - primR.u[locdim]) / (wsR - ustar);
            rhou_star = rho_star * ustar;
            E_star = rho_star * (consR.E / consR.rho + (ustar - primR.u[locdim]) * (ustar + primR.P / rcR));
        }

         EulerFlux flux;

        if (wsL * wsR > 0)
        {
            flux.rho = FluxHLLC(cons_state.rho, rho_star, flux_state.rho, 0);
            for (int idim = 0; idim < ndim; ++idim)
            {
                flux.rhou[idim] = FluxHLLC(cons_state.rhou[idim], rhou_star, flux_state.rhou[idim], 0);
            }
            flux.E = FluxHLLC(cons_state.E, E_star, flux_state.E, 0);
        }
        else
        {
            flux.rho = FluxHLLC(cons_state.rho, rho_star, flux_state.rho, S);
            for (int idim = 0; idim < ndim; ++idim)
            {
                flux.rhou[idim] = FluxHLLC(cons_state.rhou[idim], rhou_star, flux_state.rhou[idim], S);
            }
            flux.E = FluxHLLC(cons_state.E, E_star, flux_state.E, S);
        }
        return flux;
    }

    KOKKOS_FORCEINLINE_FUNCTION
    static double FluxHLLC(
            double const U,
            double const Ustar,
            double const F,
            double const ws) noexcept
    {
        return F + ws * (Ustar - U);
    }
};

} // namespace novapp
