/**
 * @file flux.hpp
 * Flux for the primary variables
 */
#pragma once

//! Flux formule
//! @param[in] rho double density
//! @param[in] u double speed
//! @param[in] P double pressure
//! @return flux
class Flux
{
private :
    double m_rho;
    double m_u;
    double m_P;

public :
      Flux(
          double const rho,
          double const u,
          double const P);
    double FluxRho();
    double FluxRhou();
    double FluxE();
};
