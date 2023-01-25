/**
 * @file float_conversion.hpp
 * Variable conversion
 */
#pragma once

//! Conversion primary variables to conservative
//! @param[in] rho density
//! @param[in] u speed
//! @param[in] P pressure
//! @param[in] gamma gamma
//! @return rhou momentum
//! @return E total energy
class ConvPtoC
{
private :
    double m_rho;
    double m_u;
    double m_P;
    double m_gamma;

public :
    ConvPtoC(double const rho,
        double const u,
        double const P,
        double const gamma)
        : m_rho(rho)
        , m_u(u)
        , m_P(P)
        , m_gamma(gamma)
    {
    }

    double ConvRhou()
    {
        return m_rho * m_u;
    }

    double ConvE()
    {
        return (1. / 2) * m_rho * m_u * m_u + m_P / (m_gamma - 1); 
    }
};

//! Conversion conservtaive variables to primary
//! @param[in] rho density
//! @param[in] E total energy
//! @param[in] P pressure
//! @param[in] gamma gamma
//! @return u speed
//! @return rhou momentum
class ConvCtoP
{
private :
    double m_rho;
    double m_rhou;
    double m_E;
    double m_gamma;
    double m_u;
    
public :
    ConvCtoP(double const rho,
        double const rhou,
        double const E, 
        double const gamma)
        : m_rho(rho)
        , m_rhou(rhou)
        , m_E(E)
        , m_gamma(gamma)
        , m_u(rhou / rho)
    {
    }

    double ConvU()
    {
        return m_rhou / m_rho;
    }

    double ConvP()
    {
        return (m_gamma - 1) * (m_E - (1. / 2) * m_rho * m_u * m_u);
    }
};