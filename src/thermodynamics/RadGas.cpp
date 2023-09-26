#include <stdexcept>

#include <Kokkos_Core.hpp>

#include "RadGas.hpp"

namespace novapp::thermodynamics
{

RadGas::RadGas(double const gamma, double const mmw)
    : m_gamma(gamma)
    , m_gamma_m1(gamma - 1)
    , m_mmw(mmw)
{
    if (!Kokkos::isfinite(gamma) || gamma <= 1)
    {
        throw std::domain_error("Invalid gamma");
    }

    if (!Kokkos::isfinite(mmw) || mmw <= 0)
    {
        throw std::domain_error("Invalid mmw");
    }
}

} // namespace novapp::thermodynamics
