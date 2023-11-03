//!
//! @file user_step.cpp
//!

#include <stdexcept>

#include <kokkos_shortcut.hpp>
#include <range.hpp>
#include <units.hpp>

#include "user_step.hpp"

namespace novapp
{

IUserStep::IUserStep() = default;

IUserStep::IUserStep([[maybe_unused]] IUserStep const& rhs) = default;

IUserStep::IUserStep([[maybe_unused]] IUserStep&& rhs) noexcept = default;

IUserStep::~IUserStep() noexcept = default;

IUserStep& IUserStep::operator=([[maybe_unused]] IUserStep const& rhs) = default;

IUserStep& IUserStep::operator=([[maybe_unused]] IUserStep&& rhs) noexcept = default;

void IUserStep::execute(
    [[maybe_unused]] Range const &range,
    [[maybe_unused]] double const t,
    [[maybe_unused]] double const dt,
    [[maybe_unused]] KV_double_3d rho,
    [[maybe_unused]] KV_double_3d E,
    [[maybe_unused]] KV_double_4d fx) const
{
    throw std::runtime_error("User step not implemented");
}


void NoUserStep::execute(
        [[maybe_unused]] Range const &range,
        [[maybe_unused]] double const t,
        [[maybe_unused]] double const dt,
        [[maybe_unused]] KV_double_3d rho,
        [[maybe_unused]] KV_double_3d E,
        [[maybe_unused]] KV_double_4d fx) const
{
}


void HeatNickelStep::execute(
    Range const &range,
    double const t,
    double const dt,
    KV_double_3d rho,
    KV_double_3d E,
    KV_double_4d fx) const
{
    double tau_ni56 = 8.8 * units::day; // s
    double tau_co56 = 111.3 * units::day;
    double m_ni56 = 55.9421278 * units::atomic_mass_unit; // kg

    double Q_ni56_tot = 1.75 * units::MeV; // J
    double Q_co56_tot = 3.73 * units::MeV;

    double fac_decay = 1. / (m_ni56 * (tau_co56 - tau_ni56)); // Decay factor

    double epsilon = (Q_ni56_tot * (tau_co56 / tau_ni56 - 1) - Q_co56_tot)
                    * Kokkos::exp(- t / tau_ni56)
                    + Q_co56_tot *  Kokkos::exp(- t / tau_co56); // Total energy rate created
    epsilon *= fac_decay;

    auto const [begin, end] = cell_range(range);
    Kokkos::parallel_for(
        "Heating_Ni56",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
        KOKKOS_LAMBDA(int i, int j, int k)
        {
            double edep = fx(i, j, k, 0) * rho(i, j, k) * epsilon;
            E(i, j, k) += edep * dt;
        });
}

} // namespace novapp
