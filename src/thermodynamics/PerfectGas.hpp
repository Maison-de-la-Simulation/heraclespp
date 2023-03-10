//!
//! @file PerfectGas.hpp
//!
#pragma once

#include <Kokkos_Core.hpp>

namespace novapp::thermodynamics
{

class PerfectGas
{
    double m_gamma;
    double m_gamma_m1;
    double m_inv_gamma_m1;
    double m_mmw;

public:
    PerfectGas(double gamma, double mmw);

    PerfectGas(const PerfectGas& rhs) = default;

    PerfectGas(PerfectGas&& rhs) noexcept = default;

    ~PerfectGas() noexcept = default;

    PerfectGas& operator=(const PerfectGas& rhs) = default;

    PerfectGas& operator=(PerfectGas&& rhs) noexcept = default;

    KOKKOS_FORCEINLINE_FUNCTION
    double compute_adiabatic_index() const noexcept
    {
        return m_gamma;
    }

    KOKKOS_FORCEINLINE_FUNCTION
    double compute_volumic_internal_energy([[maybe_unused]] double const density, double const pressure) const noexcept
    {
        return m_inv_gamma_m1 * pressure;
    }

    KOKKOS_FORCEINLINE_FUNCTION
    double compute_mean_molecular_weight() const noexcept
    {
        return m_mmw;
    }

    KOKKOS_FORCEINLINE_FUNCTION
    double compute_pressure([[maybe_unused]] double const density, double const volumic_internal_energy) const noexcept
    {
        return m_gamma_m1 * volumic_internal_energy;
    }

    KOKKOS_FORCEINLINE_FUNCTION
    double compute_speed_of_sound(double const density, double const pressure) const noexcept
    {
        return Kokkos::sqrt(m_gamma * pressure / density);
    }

    KOKKOS_FORCEINLINE_FUNCTION
    static bool is_valid(double const density, double const pressure) noexcept
    {
        return Kokkos::isfinite(density) && density > 0 && Kokkos::isfinite(pressure) && pressure > 0;
    }
};

} // namespace novapp::thermodynamics
