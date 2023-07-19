//!
//! @file riemann_solver.hpp
//!

#pragma once

#include <Kokkos_Core.hpp>

#include "euler_equations.hpp"
#include "eos.hpp"

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
NOVA_FORCEUNROLL
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

} // namespace novapp
