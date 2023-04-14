#include "PerfectGas.hpp"

namespace novapp::thermodynamics
{

PerfectGas::PerfectGas(double const gamma, double const mmw, double const T)
    : m_gamma(gamma)
    , m_gamma_m1(gamma - 1)
    , m_inv_gamma_m1(1. / (gamma - 1))
    , m_mmw(mmw)
    , m_T(T)
{
    if (!std::isfinite(gamma) || gamma <= 1)
    {
        throw std::domain_error("Invalid gamma");
    }

    if (!std::isfinite(mmw) || mmw <= 0)
    {
        throw std::domain_error("Invalid mmw");
    }
}

} // namespace novapp::thermodynamics
