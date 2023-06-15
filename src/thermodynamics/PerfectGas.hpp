//!
//! @file PerfectGas.hpp
//!

#pragma once

#include <Kokkos_Core.hpp>

#include "../units.hpp"

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

    KOKKOS_FORCEINLINE_FUNCTION
    double adiabatic_index() const noexcept
    {
        return m_gamma;
    }

    KOKKOS_FORCEINLINE_FUNCTION
    double mean_molecular_weight() const noexcept
    {
        return m_mmw;
    }

    KOKKOS_FORCEINLINE_FUNCTION
    double compute_evol_from_P(double const rho, double const P) const noexcept
    {
        return compute_evol_from_T(rho, compute_T_from_P(rho, P));
    }

    KOKKOS_FORCEINLINE_FUNCTION
    double compute_evol_from_T(double const rho, double const T) const noexcept
    {
        return rho * units::kb * T / (m_mmw * units::mh * m_gamma_m1);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    double compute_P_from_evol(double const rho, double const evol) const noexcept
    {
        return compute_P_from_T(rho, compute_T_from_evol(rho, evol));
    }

    KOKKOS_FORCEINLINE_FUNCTION
    double compute_P_from_T(double const rho, double const T) const noexcept
    {
        return rho * units::kb * T / (m_mmw * units::mh);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    double compute_T_from_P(double const rho, double const P) const noexcept
    {
        return P * m_mmw * units::mh / (rho * units::kb);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    double compute_T_from_evol(double const rho, double const evol) const noexcept
    {
        return evol * m_mmw * units::mh * m_gamma_m1 / (rho * units::kb);
    }

    KOKKOS_FORCEINLINE_FUNCTION
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
