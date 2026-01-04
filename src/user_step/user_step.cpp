// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file user_step.cpp
//!

#include <string>

#include <Kokkos_Core.hpp>
#include <kokkos_shortcut.hpp>
#include <range.hpp>
#include <units.hpp>

#include "user_step.hpp"

namespace hclpp {

IUserStep::IUserStep() = default;

IUserStep::IUserStep(IUserStep const& rhs) = default;

IUserStep::IUserStep(IUserStep&& rhs) noexcept = default;

IUserStep::~IUserStep() noexcept = default;

auto IUserStep::operator=(IUserStep const& /*rhs*/) -> IUserStep& = default;

auto IUserStep::operator=(IUserStep&& /*rhs*/) noexcept -> IUserStep& = default;

void NoUserStep::execute(
        Range const& /*range*/,
        double const /*t*/,
        double const /*dt*/,
        KV_double_3d const& /*rho*/,
        KV_double_3d const& /*E*/,
        KV_double_4d const& /*fx*/) const
{
}

void HeatNickelStep::execute(
        Range const& range,
        double const t,
        double const dt,
        KV_double_3d const& rho,
        KV_double_3d const& E,
        KV_double_4d const& fx) const
{
    double const tau_ni56 = 8.8 * units::day; // s
    double const tau_co56 = 111.3 * units::day;
    double const m_ni56 = 55.9421278 * units::atomic_mass_unit; // kg

    double const Q_ni56_tot = 1.75 * units::MeV; // J
    double const Q_co56_tot = 3.73 * units::MeV;

    double const fac_decay = 1. / (m_ni56 * (tau_co56 - tau_ni56)); // Decay factor

    double epsilon = ((Q_ni56_tot * (tau_co56 / tau_ni56 - 1) - Q_co56_tot) * Kokkos::exp(-t / tau_ni56))
                     + (Q_co56_tot * Kokkos::exp(-t / tau_co56)); // Total energy rate created
    epsilon *= fac_decay;

    Kokkos::parallel_for(
            "Heating_Ni56",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k) {
                double const edep = fx(i, j, k, 0) * rho(i, j, k) * epsilon;
                E(i, j, k) += edep * dt;
            });
}

} // namespace hclpp
