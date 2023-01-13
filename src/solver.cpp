#include <Kokkos_Core.hpp>

#include "global_var.hpp"
#include "solver.hpp"
#include "conv.hpp"
#include "flux.hpp"

using namespace GV;

double sound_speed(
        double rhok,
        double Pk)
{
    return std::sqrt(GV::gamma * Pk / rhok);
}

WaveSpeed::WaveSpeed(
        double const rhoL,
        double const uL,
        double const PL,
        double const rhoR,
        double const uR,
        double const PR)
{
        m_rhoL = rhoL;
        m_uL = uL;
        m_PL = PL;
        m_rhoR = rhoR;
        m_uR = uR;
        m_PR = PR;
        m_cL = sound_speed(rhoL, PL);
        m_cR = sound_speed(rhoR, PR);
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
        double const ER)
{
        m_rhoL = rhoL;
        m_rhouL = rhouL;
        m_EL = EL;
        m_rhoR = rhoR;
        m_rhouR = rhouR;
        m_ER = ER;

        ConvCtoP convCtoPL(m_rhoL, m_rhouL, m_EL);
        m_uL = convCtoPL.ConvU();
        m_PL = convCtoPL.ConvP();
        ConvCtoP convCtoPR(m_rhoR, m_rhouR, m_ER);
        m_uR = convCtoPR.ConvU();
        m_PR = convCtoPR.ConvP();

        WaveSpeed WS(m_rhoL, m_uL, m_PL, m_rhoR, m_uR, m_PR);
        m_SL = WS.SL();
        m_SR = WS.SR();
        m_diff = m_SR - m_SL;

        Flux fluxL(m_rhoL, m_uL, m_PL);
        m_FrhoL = fluxL.FluxRho();
        m_FrhouL = fluxL.FluxRhou();
        m_FEL = fluxL.FluxE();

        Flux fluxR(m_rhoR, m_uR, m_PR);
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
        if (m_SR <= 0)
        {
            return m_FrhoR;
        }
        else
        {
            return FluxHLL(m_rhoL, m_rhoR, m_FrhoL, m_FrhoR);
        }
}

double SolverHLL::FinterRhou()
{
        if (m_SL >= 0)
        {
            return m_FrhouL;
        }
        if (m_SR <= 0)
        {
            return m_FrhouR;
        }
        else
        {
            return FluxHLL(m_rhouL, m_rhouR, m_FrhouL, m_FrhouR);
        }
}

double SolverHLL::FinterE()
{
        if (m_SL >= 0)
        {
            return m_FEL;
        }
        if (m_SR <= 0)
        {
            return m_FER;
        }
        else
        {
            return FluxHLL(m_EL, m_ER, m_FEL, m_FER);
        }
}
