/**
 * @file flux.hpp
 * Flux for the primary variables
 */
#pragma once

//! Flux formule
//! @param[in] rho double density
//! @param[in] u double speed
//! @param[in] P double pressure
//! @param[in] gamma double
//! @return flux
class Flux
{
private:
      double m_rho;
      double m_u;
      double m_P;
      double m_gamma;

public :
      Flux(
            double const rho,
            double const u,
            double const P,
            double const gamma);
      double FluxRho();
      double FluxRhou();
      double FluxE();
};
