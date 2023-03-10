//!
//! @file riemann_solver.hpp
//!
#pragma once

namespace novapp
{

class HLL
{
public:
    //! HLL solver
    //! @param[in] 
    //! @return intercell EulerFlux
    KOKKOS_INLINE_FUNCTION
    EulerFlux operator()(
            EulerCons const& consL,
            EulerCons const& consR,
            int locdim,
            thermodynamics::PerfectGas const& eos) const noexcept
    {
        EulerPrim const primL = to_prim(consL, eos);
        EulerPrim const primR = to_prim(consR, eos);

        double const cL = eos.compute_speed_of_sound(primL.density, primL.pressure);
        double const cR = eos.compute_speed_of_sound(primR.density, primR.pressure);

        double const wsL = std::fmin(primL.velocity[locdim] - cL, primR.velocity[locdim] - cR);
        double const wsR = std::fmax(primL.velocity[locdim] + cL, primR.velocity[locdim] + cR);

        EulerFlux const fluxL = compute_flux(primL, locdim, eos);
        EulerFlux const fluxR = compute_flux(primR, locdim, eos);

        if (wsL >= 0)
        {
            return fluxL;
        }

        if (wsL <= 0 && wsR >= 0)
        {
            EulerFlux flux;
            flux.density
                    = FluxHLL(consL.density, consR.density, fluxL.density, fluxR.density, wsL, wsR);
            for (int idim = 0; idim < ndim; ++idim)
            {
                flux.momentum[idim] = FluxHLL(
                    consL.momentum[idim],
                    consR.momentum[idim],
                    fluxL.momentum[idim],
                    fluxR.momentum[idim],
                    wsL,
                    wsR);
            }
            flux.energy = FluxHLL(consL.energy, consR.energy, fluxL.energy, fluxR.energy, wsL, wsR);
            return flux;
        }

        if (wsR <= 0)
        {
            return fluxR;
        }

        return EulerFlux {};
    }

    KOKKOS_INLINE_FUNCTION
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
