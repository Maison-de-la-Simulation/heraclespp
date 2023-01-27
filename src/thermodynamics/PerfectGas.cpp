#include "PerfectGas.hpp"

namespace thermodynamics
{

PerfectGas::PerfectGas(double const gamma, double const mmw) noexcept
    : m_gamma(gamma)
    , m_gamma_m1(gamma - 1)
    , m_inv_gamma_m1(1. / (gamma - 1))
    , m_mmw(mmw)
{
}

} // namespace thermodynamics
