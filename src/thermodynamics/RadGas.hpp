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
    double compute_evol_from_P(double const rho, double const P) const noexcept
    {
        return compute_evol_from_T(rho, compute_T_from_P(rho, P));
    }

    KOKKOS_FORCEINLINE_FUNCTION
    double compute_evol_from_T(double const rho, double const T) const noexcept
    {
        auto const T4 = T * T * T * T;
        return rho * units::kb * T / (m_mmw * units::mh * m_gamma_m1) + units::ar * T4;
    }

    KOKKOS_FORCEINLINE_FUNCTION
    double compute_P_from_evol(double const rho, double const evol) const noexcept
    {
        return compute_P_from_T(rho, compute_T_from_evol(rho, evol));
    }

    KOKKOS_FORCEINLINE_FUNCTION
    double compute_P_from_T(double const rho, double const T) const noexcept
    {
        auto const T4 = T * T * T * T;
        //std::cout<<"Pg = "<<rho * units::kb * T / (m_mmw * units::mh)<<" Pr = "<<units::ar * T4 / 3<<std::endl;
        return rho * units::kb * T / (m_mmw * units::mh) + units::ar * T4 / 3;
    }

    KOKKOS_FORCEINLINE_FUNCTION
    double compute_T_from_P(double const rho, double const P) const noexcept
    {
        double T;
        double C1 = rho * units::kb / (m_mmw * units::mh);
        double Tg = P / C1;
        double Tr = Kokkos::pow(3 * P / units::ar, 1. / 4);
        double T0 = Tg;
        if (Tr > Tg)
        {
            T0 = Tr;
        }
        T = T0;
        // int itr;
        int max_itr = 100;
        for (int i = 0; i < max_itr; ++i)
        {
            double T3 = T * T * T;
            double f = units::ar * T3 * T / 3 + C1 * T - P;
            double df = 4 * units::ar * T3 / 3 + C1;
            double delta_T = -f / df;
            T += delta_T;
            // itr = i;
            if (Kokkos::abs(delta_T) < 1E-6)
            {
                break;
            }
        }
        //std::cout<<"T_from_P = "<<T<<" itr = "<<itr<<std::endl;
        return T;
    }

    KOKKOS_FORCEINLINE_FUNCTION
    double compute_T_from_evol(double const rho, double const evol) const noexcept
    {
        double T;
        double C1 = rho * units::kb / (m_mmw * units::mh * m_gamma_m1);
        double Tg = evol / C1;
        double Tr = Kokkos::pow(evol / units::ar, 1. / 4);
        double T0 = Tg;
        if (Tr > Tg)
        {
            T0 = Tr;
        }
        T = T0;
        // int itr;
        int max_itr = 100;
        for (int i = 0; i < max_itr; ++i)
        {
            double T3 = T * T * T;
            double f = units::ar * T3 * T + C1 * T - evol;
            double df = 4 * units::ar * T3 + C1;
            double delta_T = -f / df;
            T += delta_T;
            // itr = i;
            if (Kokkos::abs(delta_T) < 1E-6)
            {
                break;
            }
        }
        //std::cout<<"T_from_evol = "<<T<<" itr = "<<itr<<std::endl;
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
};

} // namespace novapp::thermodynamics
