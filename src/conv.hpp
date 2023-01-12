/**
 * @file conv.hpp
 * Variable conversion
 */
#pragma once

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

class ConvCtoP
{
private :
        double m_rho;
        double m_rhou;
        double m_E;

public :
      ConvCtoP(
              double const rho,
              double const rhou,
              double const E);
      double ConvU();
      double ConvP();
};
