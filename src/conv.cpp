#include "conv.hpp"
#include "global_var.hpp"
using namespace GV;

ConvPtoC::ConvPtoC(
        double const rho,
        double const u,
        double const P)
{
        m_rho = rho;
        m_u = u;
        m_P = P;
};

double ConvPtoC::ConvRhou()
{
        return m_rho * m_u;
}

double ConvPtoC::ConvE()
{
        return (1. / 2) * m_rho * m_u * m_u + m_P / (GV::gamma - 1);
}

ConvCtoP::ConvCtoP(
        double const rho,
        double const rhou,
        double const E)
{
        m_rho = rho;
        m_rhou = rhou;
        m_E = E;
};

double ConvCtoP::ConvU()
{
        return m_rhou / m_rho;
}

double ConvCtoP::ConvP()
{
        return (GV::gamma - 1) * (m_E - (1. / 2) * m_rhou * m_rhou / m_rho);
}
