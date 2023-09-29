//!
//! @file user_step.hpp
//!

#pragma once

#include <array>
#include <stdexcept>

#include <kokkos_shortcut.hpp>
#include <ndim.hpp>
#include <nova_params.hpp>
#include <range.hpp>
#include <units.hpp>

namespace novapp
{

class I_User_Step
{
public:
    I_User_Step() = default;

    I_User_Step(I_User_Step const& rhs) = default;

    I_User_Step(I_User_Step&& rhs) noexcept = default;

    virtual ~I_User_Step() noexcept = default;

    I_User_Step& operator=(I_User_Step const& rhs) = default;

    I_User_Step& operator=(I_User_Step&& rhs) noexcept = default;

    virtual void execute(
        [[maybe_unused]] Range const &range,
        [[maybe_unused]] double const t,
        [[maybe_unused]] double const dt,
        [[maybe_unused]] KV_double_3d rho,
        [[maybe_unused]] KV_double_3d E,
        [[maybe_unused]] KV_double_4d fx) const 
        {
            throw std::runtime_error("User step not implemented");
        }
};

class No_User_Step : public I_User_Step
{

}

class Heat_56_Nickel_Step : public I_User_Step
{
    void execute(
        Range const &range,
        double const t,
        double const dt,
        KV_double_3d rho,
        KV_double_3d E,
        KV_double_4d fx)
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
};

inline std::unique_ptr<I_User_Step> factory_user_step(
    std::string const& user_step)
{
    if (user_step == "None")
    {
        return std::make_unique<No_User_Step>;
    }
    if (user_step == "Heat_nickel")
    {
        return std::make_unique<Heat_56_Nickel_Step>;
    }
    throw std::runtime_error("Unknown user step : " + user_step + ".");
}


} // namespace novapp
