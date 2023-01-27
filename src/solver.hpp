/**
 * @file solver.hpp
 * Riemann solver
 */
#pragma once

namespace thermodynamics
{

class PerfectGas;

} // namespace thermodynamics

//! Wave speed
//! @param[in] rhoL density left
//! @param[in] uL speed left
//! @param[in] PL pressure left
//! @param[in] rhoR density right
//! @param[in] uR speed right
//! @param[in] PR pressure right
//! @return SL wave speed left
//! @return SR wave speed right

class WaveSpeed
{
  private :
          double m_rhoL;
          double m_uL;
          double m_PL;
          double m_rhoR;
          double m_uR;
          double m_PR;
          double m_cL;
          double m_cR;

  public :
          WaveSpeed(
                  double const rhoL,
                  double const uL,
                  double const PL,
                  double const rhoR,
                  double const uR,
                  double const PR,
                  thermodynamics::PerfectGas const& eos);
          double SL();
          double SR();
};

class SolverHLL
{
  private :
          double m_rhoL;
          double m_rhouL;
          double m_EL;
          double m_rhoR;
          double m_rhouR;
          double m_ER;
          double m_uL;
          double m_PL;
          double m_uR;
          double m_PR;
          double m_SL;
          double m_SR;
          double m_diff;
          double m_FrhoL;
          double m_FrhouL;
          double m_FEL;
          double m_FrhoR;
          double m_FrhouR;
          double m_FER;
  public :
          SolverHLL(
                  double const rhoL,
                  double const rhouL,
                  double const EL,
                  double const rhoR,
                  double const rhouR,
                  double const ER, 
                  thermodynamics::PerfectGas const& eos);
          double FluxHLL(double UL, double UR, double FL, double FR);
          double FinterRho();
          double FinterRhou();
          double FinterE();
};
