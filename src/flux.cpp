# include "flux.hpp"
#include "global_var.hpp"
using namespace GV;

Flux::Flux(
    double const rho,
    double const u,
    double const P)
{
    m_rho = rho;
    m_u = u;
    m_P = P;
};

double Flux::FluxRho()
{
    return m_rho * m_u;
}

double Flux::FluxRhou()
{
    return m_rho * m_u * m_u + m_P;
}

double Flux::FluxE()
{
    return m_u * ((1. / 2) * m_rho * m_u * m_u + m_P / (GV::gamma - 1) + m_P);
}
