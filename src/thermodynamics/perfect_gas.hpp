// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file perfect_gas.hpp
//!

#pragma once

#include <Kokkos_Macros.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <units.hpp>

namespace novapp::thermodynamics
{

class PerfectGas
{
    double m_gamma;
    double m_gamma_m1;
    double m_mmw;

public:
    PerfectGas(double gamma, double mmw);

    PerfectGas(const PerfectGas& rhs) = default;

    PerfectGas(PerfectGas&& rhs) noexcept = default;

    ~PerfectGas() noexcept = default;

    PerfectGas& operator=(const PerfectGas& rhs) = default;

    PerfectGas& operator=(PerfectGas&& rhs) noexcept = default;

    [[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION
    double adiabatic_index() const noexcept
    {
        return m_gamma;
    }

    [[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION
    double mean_molecular_weight() const noexcept
    {
        return m_mmw;
    }

    [[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION
    double compute_evol_from_pres(double const rho, double const P) const noexcept
    {
        // evol = rho * eint
        return compute_evol_from_temp(rho, compute_temp_from_pres(rho, P));
    }

    [[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION
    double compute_evol_from_temp(double const rho, double const T) const noexcept
    {
        // evol = rho * eint
        return rho * units::kb * T / (m_mmw * units::mp * m_gamma_m1);
    }

    [[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION
    double compute_pres_from_evol(double const rho, double const evol) const noexcept
    {
        // evol = rho * eint
        return compute_pres_from_temp(rho, compute_temp_from_evol(rho, evol));
    }

    [[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION
    double compute_pres_from_temp(double const rho, double const T) const noexcept
    {
        return rho * units::kb * T / (m_mmw * units::mp);
    }

    [[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION
    double compute_temp_from_pres(double const rho, double const P) const noexcept
    {
        return P * m_mmw * units::mp / (rho * units::kb);
    }

    [[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION
    double compute_temp_from_evol(double const rho, double const evol) const noexcept
    {
        // evol = rho * eint
        return evol * m_mmw * units::mp * m_gamma_m1 / (rho * units::kb);
    }

    [[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION
    double compute_speed_of_sound(double const rho, double const P) const noexcept
    {
        return Kokkos::sqrt(m_gamma * P / rho);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    static bool is_valid(double const rho, double const P) noexcept
    {
        return Kokkos::isfinite(rho) && rho > 0 && Kokkos::isfinite(P) && P > 0;
    }
};

} // namespace novapp::thermodynamics
