#include <Kokkos_Core.hpp>

#include "solver.hpp"
#include "float_conversion.hpp"
#include "flux.hpp"
#include "speed_sound.hpp"

WaveSpeed::WaveSpeed(
    double const rhoL,
    double const uL,
    double const PL,
    double const rhoR,
    double const uR,
    double const PR,
    double const gamma)
{
    m_rhoL = rhoL;
    m_uL = uL;
    m_PL = PL;
    m_rhoR = rhoR;
    m_uR = uR;
    m_PR = PR;
    m_gamma = gamma;
    m_cL = speed_sound2(rhoL, PL, gamma);
    m_cR = speed_sound2(rhoR, PR, gamma);
};

double WaveSpeed::SL()
{
    return std::min(m_uL - m_cL, m_uR - m_cR);
}
double WaveSpeed::SR()
{
    return std::max(m_uL + m_cL, m_uR + m_cR);
}

SolverHLL::SolverHLL(
    double const rhoL,
    double const rhouL,
    double const EL,
    double const rhoR,
    double const rhouR,
    double const ER,
    double const gamma)
{
    m_rhoL = rhoL;
    m_rhouL = rhouL;
    m_EL = EL;
    m_rhoR = rhoR;
    m_rhouR = rhouR;
    m_ER = ER;
    m_gamma = gamma;

    ConvCtoP convCtoPL(m_rhoL, m_rhouL, m_EL, m_gamma);
    m_uL = convCtoPL.ConvU();
    m_PL = convCtoPL.ConvP();
    ConvCtoP convCtoPR(m_rhoR, m_rhouR, m_ER, m_gamma);
    m_uR = convCtoPR.ConvU();
    m_PR = convCtoPR.ConvP();

    WaveSpeed WS(m_rhoL, m_uL, m_PL, m_rhoR, m_uR, m_PR, m_gamma);
    m_SL = WS.SL();
    m_SR = WS.SR();
    m_diff = m_SR - m_SL;

    Flux fluxL(m_rhoL, m_uL, m_PL, m_gamma);
    m_FrhoL = fluxL.FluxRho();
    m_FrhouL = fluxL.FluxRhou();
    m_FEL = fluxL.FluxE();

    Flux fluxR(m_rhoR, m_uR, m_PR, m_gamma);
    m_FrhoR = fluxR.FluxRho();
    m_FrhouR = fluxR.FluxRhou();
    m_FER = fluxR.FluxE();
};

double SolverHLL::FluxHLL(double UL, double UR, double FL, double FR)
{
    return (m_SR * FL - m_SL * FR + m_SL * m_SR * (UR - UL)) / m_diff;
}

double SolverHLL::FinterRho()
{
    if (m_SL >= 0)
    {
        return m_FrhoL;
    }
    if (m_SL <= 0 && m_SR >= 0)
    {
        return FluxHLL(m_rhoL, m_rhoR, m_FrhoL, m_FrhoR);
    }
    if (m_SR <= 0)
    {
        return m_FrhoR;
    }  
}

double SolverHLL::FinterRhou()
{
    if (m_SL >= 0)
    {
        return m_FrhouL;
    }
    if (m_SL <= 0 && m_SR >= 0)
    {
        return FluxHLL(m_rhouL, m_rhouR, m_FrhouL, m_FrhouR);
    }
    if (m_SR <= 0)
    {
        return m_FrhouR;
    }
}

double SolverHLL::FinterE()
{
    if (m_SL >= 0)
    {
        return m_FEL;
    }
    if (m_SL <= 0 && m_SR >= 0)
    {
        return FluxHLL(m_EL, m_ER, m_FEL, m_FER);
    }
    if (m_SR <= 0)
    {
        return m_FER;
    }
}
