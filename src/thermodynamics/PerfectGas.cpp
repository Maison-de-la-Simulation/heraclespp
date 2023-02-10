#include "PerfectGas.hpp"

namespace thermodynamics
{

PerfectGas::PerfectGas(double const gamma, double const mmw)
    : m_gamma(gamma)
    , m_gamma_m1(gamma - 1)
    , m_inv_gamma_m1(1. / (gamma - 1))
    , m_mmw(mmw)
{
    if (!(gamma > 1))
    {
        throw std::runtime_error("Invalid gamma");
    }

    if (!(mmw > 0))
    {
        throw std::runtime_error("Invalid mmw");
    }
}

} // namespace thermodynamics
