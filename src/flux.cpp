# include "flux.hpp"

Flux::Flux(
        double const rho,
        double const u,
        double const P,
        double const gamma)
{
        m_rho = rho;
        m_u = u;
        m_P = P;
        m_gamma = gamma;
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
    return m_u * (1. / 2) * m_rho * m_u * m_u + m_P / (m_gamma - 1) + m_P;
}
