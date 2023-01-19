/**
 * @file solver.hpp
 * Riemann solver
 */
#pragma once

//! Sound speed
//! @param[in] rhok density with k = left or right
//! @param[in] Pk pressure with k = left or right
//! @return sound speed
double sound_speed(
        double rhok,
        double Pk);

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
          double m_gamma;
          double m_cL;
          double m_cR;

  public :
          WaveSpeed(
                  double const rhoL,
                  double const uL,
                  double const PL,
                  double const rhoR,
                  double const uR,
                  double const PR);
          double SL();
          double SR();
};

double Dt(Kokkos::View<double*> const rhoL,
    Kokkos::View<double*> const uL,
    Kokkos::View<double*> const PL,
    Kokkos::View<double*> const rhoR,
    Kokkos::View<double*> const uR,
    Kokkos::View<double*> const PR,
    double const dx, 
    double const cfl);

//! HLL solver
//! @param[in] rhoL density left
//! @param[in] uL speed left
//! @param[in] PL pressure left
//! @param[in] rhoR density right
//! @param[in] uR speed right
//! @param[in] PR pressure right
//! @return intercell flux density
//! @return intercell flux momentum
//! @return intercell flux energy
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
                  double const ER);
          double FluxHLL(double UL, double UR, double FL, double FR);
          double FinterRho();
          double FinterRhou();
          double FinterE();
};
