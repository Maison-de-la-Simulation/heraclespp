/**
 * @file conv.hpp
 * Variable conversion
 */
#pragma once

//! Conversion primary variables to conservative
//! @param[in] rho density
//! @param[in] u speed
//! @param[in] P pressure
//! @return rhou momentum
//! @return E total energy
class ConvPtoC
{
private :
        double m_rho;
        double m_u;
        double m_P;

public :
      ConvPtoC(
              double const rho,
              double const u,
              double const P);
      double ConvRhou();
      double ConvE();
};

//! Conversion conservtaive variables to primary
//! @param[in] rho density
//! @param[in] E total energy
//! @param[in] P pressure
//! @return u speed
//! @return rhou momentum
class ConvCtoP
{
private :
        double m_rho;
        double m_rhou;
        double m_E;
        double m_u;

public :
      ConvCtoP(
              double const rho,
              double const rhou,
              double const E);
      double ConvU();
      double ConvP();
};
