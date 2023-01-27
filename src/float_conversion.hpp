/**
 * @file float_conversion.hpp
 * Variable conversion
 */
#pragma once

#include <PerfectGas.hpp>

//! Conversion primary variables to conservative
//! @param[in] rho density
//! @param[in] u speed
//! @param[in] P pressure
//! @param[in] eos equation of state
//! @return rhou momentum
//! @return E total energy
class ConvPtoC
{
private :
    double m_rho;
    double m_u;
    double m_P;
    thermodynamics::PerfectGas const& m_eos;

public :
    ConvPtoC(double const rho,
        double const u,
        double const P,
        thermodynamics::PerfectGas const& eos)
        : m_rho(rho)
        , m_u(u)
        , m_P(P)
        , m_eos(eos)
    {
    }

    double ConvRhou()
    {
        return m_rho * m_u;
    }

    double ConvE()
    {
        return (1. / 2) * m_rho * m_u * m_u + m_eos.compute_volumic_internal_energy(m_rho, m_P);
    }
};

//! Conversion conservtaive variables to primary
//! @param[in] rho density
//! @param[in] E total energy
//! @param[in] P pressure
//! @param[in] eos equation of state
//! @return u speed
//! @return rhou momentum
class ConvCtoP
{
private :
    double m_rho;
    double m_rhou;
    double m_E;
    double m_u;
    thermodynamics::PerfectGas const& m_eos;
    
public :
    ConvCtoP(double const rho,
        double const rhou,
        double const E, 
        thermodynamics::PerfectGas const& eos)
        : m_rho(rho)
        , m_rhou(rhou)
        , m_E(E)
        , m_u(rhou / rho)
        , m_eos(eos)
    {
    }

    double ConvU()
    {
        return m_rhou / m_rho;
    }

    double ConvP()
    {
        return m_eos.compute_pressure(m_rho, m_E - (1. / 2) * m_rho * m_u * m_u);
    }
};
