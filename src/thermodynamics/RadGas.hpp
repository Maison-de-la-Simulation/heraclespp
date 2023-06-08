//!
//! @file RadGas.hpp
//!

#pragma once

#include <Kokkos_Core.hpp>

#include "../units.hpp"

namespace novapp::thermodynamics
{

class RadGas
{
    [[maybe_unused]] double m_gamma;
    [[maybe_unused]] double m_gamma_m1;
    [[maybe_unused]] double m_inv_gamma_m1;
    [[maybe_unused]] double m_mmw;

public:
    RadGas(double gamma, double mmw);

    RadGas(const RadGas& rhs) = default;

    RadGas(RadGas&& rhs) noexcept = default;

    ~RadGas() noexcept = default;

    RadGas& operator=(const RadGas& rhs) = default;

    RadGas& operator=(RadGas&& rhs) noexcept = default;
     
   /* KOKKOS_FORCEINLINE_FUNCTION
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
    double compute_evol_from_T(double const rho, double const T) const noexcept
    {
        auto const T4 = T * T * T * T;
        return rho * units::kb * T / (m_mmw * units::mh) * m_inv_gamma_m1 + units::ar * T4;
    }

    KOKKOS_FORCEINLINE_FUNCTION
    double compute_P_from_T(double const rho, double const T) const noexcept
    {
        auto const T4 = T * T * T * T;
        return rho * units::kb * T / (m_mmw * units::mh) + units::ar * T4 / 3;
    }

    KOKKOS_FORCEINLINE_FUNCTION
    double compute_T_from_P(double const rho, double const P) const noexcept
    {
        double T;
        double C1 = rho * units::kb / (m_mmw * units::mh);
        double Tg = P * m_mmw * units::mh / (rho * units::kb);
        double Tr = std::pow(3 * P / units::ar, 1. / 4);
        double T0 = Tg;
        if (Tr > Tg)
        {
            T0 = Tr;
        }
        T = T0;
        double delta_T;
        int n = 0;
        while (n < 100 && std::abs(delta_T) >= 1E-6)
        {
            double T3 = T * T * T;
            double f = units::ar * T3 * T / 3 + C1 * T - P;
            double df = 4 * units::ar * T3 + C1;
            delta_T = -f / df;
            T += delta_T;
            ++n;
        }
        return T;
    }

    KOKKOS_FORCEINLINE_FUNCTION
    double compute_T_from_evol(double const rho, double const evol) const noexcept
    {
        double T;
        double C1 = rho * units::kb / (m_mmw * units::mh) * m_inv_gamma_m1;
        double Tg = evol * m_mmw * units::mh * m_gamma_m1 / (rho * units::kb);
        double Tr = std::pow(evol / units::ar, 1. / 4);
        double T0 = Tg;
        if (Tr > Tg)
        {
            T0 = Tr;
        }
        T = T0;
        double delta_T;
        int n = 0;
        while (n < 100 && std::abs(delta_T) >= 1E-6)
        {
            double T3 = T * T * T;
            double f = units::ar * T3 * T + C1 * T - evol;
            double df = 4 * units::ar * T3 + C1;
            delta_T = -f / df;
            T += delta_T;
            ++n;
        }
        return T;
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
 */
};

} // namespace novapp::thermodynamics
