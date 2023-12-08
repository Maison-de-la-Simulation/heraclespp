//!
//! @file RadGas.hpp
//!

#pragma once

#include <Kokkos_Core.hpp>
#include <units.hpp>

namespace novapp::thermodynamics
{

class RadGas
{
    double m_gamma;
    double m_gamma_m1;
    double m_mmw;

public:
    RadGas(double gamma, double mmw);

    RadGas(const RadGas& rhs) = default;

    RadGas(RadGas&& rhs) noexcept = default;

    ~RadGas() noexcept = default;

    RadGas& operator=(const RadGas& rhs) = default;

    RadGas& operator=(RadGas&& rhs) noexcept = default;

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
    double compute_eint_from_P(double const rho, double const P) const noexcept
    {
        return compute_eint_from_T(rho, compute_T_from_P(rho, P));
    }

    KOKKOS_FORCEINLINE_FUNCTION
    double compute_eint_from_T(double const rho, double const T) const noexcept
    {
        auto const T4 = T * T * T * T;
        return rho * units::kb * T / (m_mmw * units::mp * m_gamma_m1) + units::ar * T4;
    }

    KOKKOS_FORCEINLINE_FUNCTION
    double compute_P_from_eint(double const rho, double const eint) const noexcept
    {
        return compute_P_from_T(rho, compute_T_from_eint(rho, eint));
    }

    KOKKOS_FORCEINLINE_FUNCTION
    double compute_P_from_T(double const rho, double const T) const noexcept
    {
        auto const T4 = T * T * T * T;
        //std::cout<<"Pg = "<<rho * units::kb * T / (m_mmw * units::mp)<<" Pr = "<<units::ar * T4 / 3<<std::endl;
        return rho * units::kb * T / (m_mmw * units::mp) + units::ar * T4 / 3;
    }

    KOKKOS_FORCEINLINE_FUNCTION
    double compute_T_from_P(double const rho, double const P) const noexcept
    {
        double T;
        double C1 = rho * units::kb / (m_mmw * units::mp);
        double Tg = P / C1;
        double Tr = Kokkos::pow(3 * P / units::ar, 1. / 4);
        double T0 = Tg;
        if (Tr > Tg)
        {
            T0 = Tr;
        }
        T = T0;
        int static constexpr max_itr = 100;
        int itr = 0;
        double delta_T = 0;
        for (int i = 0; i < max_itr; ++i)
        {
            double T3 = T * T * T;
            double f = units::ar * T3 * T / 3 + C1 * T - P;
            double df = 4 * units::ar * T3 / 3 + C1;
            delta_T = -f / df;
            T += delta_T;
            itr = i;
            if (Kokkos::abs(delta_T) <= 1E-6)
            {
                break;
            }
        }
        if (itr == 99 && Kokkos::abs(delta_T) > 1E-6)
        {
            Kokkos::printf("No convergence in temperature : %f \n", Kokkos::abs(delta_T));
        }
        return T;
    }

    KOKKOS_FORCEINLINE_FUNCTION
    double compute_T_from_eint(double const rho, double const eint) const noexcept
    {
        double T;
        double C1 = rho * units::kb / (m_mmw * units::mp * m_gamma_m1);
        double Tg = eint / C1;
        double Tr = Kokkos::pow(eint / units::ar, 1. / 4);
        double T0 = Tg;
        if (Tr > Tg)
        {
            T0 = Tr;
        }
        T = T0;
        int static constexpr max_itr = 100;
        int itr = 0;
        double delta_T = 0;
        for (int i = 0; i < max_itr; ++i)
        {
            double T3 = T * T * T;
            double f = units::ar * T3 * T + C1 * T - eint;
            double df = 4 * units::ar * T3 + C1;
            delta_T = -f / df;
            T += delta_T;
            itr = i;
            if (Kokkos::abs(delta_T) <= 1E-6)
            {
                break;
            }
        }
        if (itr == 99 && Kokkos::abs(delta_T) > 1E-6)
        {
            Kokkos::printf("No convergence in temperature : %f \n", Kokkos::abs(delta_T));
        }
        return T;
    }

    KOKKOS_FORCEINLINE_FUNCTION
    double compute_speed_of_sound(double const rho, double const P) const noexcept
    {
        double T = compute_T_from_P(rho, P);
        double Pg = rho * units::kb * T / (m_mmw * units::mp);
        auto const T4 = T * T * T * T;
        double Pr = units::ar * T4 / 3;
        double alpha = Pr / Pg;
        double num = m_gamma / (m_gamma - 1) + 20 * alpha + 16 * alpha * alpha;
        double den = 1. / (m_gamma - 1) + 12 * alpha;
        double gamma_eff = num / den;
        return Kokkos::sqrt(gamma_eff * P / rho);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    static bool is_valid(double const rho, double const P) noexcept
    {
        return Kokkos::isfinite(rho) && rho > 0 && Kokkos::isfinite(P) && P > 0;
    }
};

} // namespace novapp::thermodynamics
