// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <stdexcept>

#include <Kokkos_MathematicalFunctions.hpp>

#include "PerfectGas.hpp"

namespace novapp::thermodynamics
{

PerfectGas::PerfectGas(double const gamma, double const mmw)
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
