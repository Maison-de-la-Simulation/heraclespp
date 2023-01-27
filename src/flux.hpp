/**
 * @file flux.hpp
 * Flux for the primary variables
 */
#pragma once

#include <PerfectGas.hpp>

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
    thermodynamics::PerfectGas const& m_eos;

public :
    Flux(double const rho,
        double const u,
        double const P,
        thermodynamics::PerfectGas const& eos)
        : m_rho(rho)
        , m_u(u)
        , m_P(P)
        , m_eos(eos)
    {
    }

    double FluxRho()
    {
        return m_rho * m_u;
    }

    double FluxRhou()
    {
        return m_rho * m_u * m_u + m_P;
    }

    double FluxE()
    {
        return m_u * ((1. / 2) * m_rho * m_u * m_u + m_eos.compute_volumic_internal_energy(m_rho, m_P) + m_P);
    }
};
