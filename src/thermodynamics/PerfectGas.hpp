//!
//! @file PerfectGas.hpp
//!
#pragma once

#include <cmath>

#include <Kokkos_Core.hpp>

namespace thermodynamics
{

//!
//! @class PerfectGas
//! @brief This class represents a perfect i.e. following:
//! @f$p = (\gamma - 1) \rho e@f$. Domain of validity
//! is @f$\mathcal{D} = \{\rho > 0\}@f$.
//!
class PerfectGas
{
    double m_gamma;

    double m_gamma_m1;

    double m_inv_gamma_m1;

    double m_mmw;

public:
    //!
    //! @fn PerfectGas (double gamma, double mmw) noexcept
    //! @brief Constructs a perfect gas.
    //! @param[in] gamma Adiabatic index of the perfect gas.
    //! @param[in] mmw Mean molecular weight of the perfect gas.
    //!
    PerfectGas(double gamma, double mmw) noexcept;

    //!
    //! @fn PerfectGas(const PerfectGas& rhs)
    //!
    PerfectGas(const PerfectGas& rhs) = default;

    //!
    //! @fn PerfectGas(PerfectGas&& rhs)
    //!
    PerfectGas(PerfectGas&& rhs) noexcept = default;

    //!
    //! @fn ~PerfectGas()
    //!
    ~PerfectGas() noexcept = default;

    //!
    //! @fn PerfectGas& operator=(const PerfectGas& rhs)
    //!
    PerfectGas& operator=(const PerfectGas& rhs) = default;

    //!
    //! @fn PerfectGas& operator=(PerfectGas&& rhs)
    //!
    PerfectGas& operator=(PerfectGas&& rhs) noexcept = default;

    //!
    //! @fn double compute_adiabatic_index() const noexcept
    //! @return Adiabatic index.
    //!
    KOKKOS_FORCEINLINE_FUNCTION
    double compute_adiabatic_index() const noexcept
    {
        return m_gamma;
    }

    //!
    //! @fn double compute_internal_energy(double density, double p) const noexcept
    //! @param[in] density Density.
    //! @param[in] pressure Pressure.
    //! @return Internal energy.
    //!
    KOKKOS_FORCEINLINE_FUNCTION
    double compute_volumic_internal_energy([[maybe_unused]] double const density, double const pressure) const noexcept
    {
        return m_inv_gamma_m1 * pressure;
    }

    //!
    //! @fn double compute_mean_molecular_weight() const noexcept
    //! @return Mean molecular weight.
    //!
    KOKKOS_FORCEINLINE_FUNCTION
    double compute_mean_molecular_weight() const noexcept
    {
        return m_mmw;
    }

    //!
    //! @fn double compute_pressure(double const density, double const volumic_internal_energy) const noexcept
    //! @param[in] density Density.
    //! @param[in] volumic_internal_energy Volumic internal energy.
    //! @return Pressure.
    //!
    KOKKOS_FORCEINLINE_FUNCTION
    double compute_pressure([[maybe_unused]] double const density, double const volumic_internal_energy) const noexcept
    {
        return m_gamma_m1 * volumic_internal_energy;
    }

    //!
    //! @fn double compute_speed_of_sound(double density, double p) const noexcept
    //! @brief @f$ c = \sqrt{\gamma \frac{p}{\rho}}@f$.
    //! @param[in] d Density.
    //! @param[in] p Pressure.
    //! @return Speed of sound.
    //!
    KOKKOS_FORCEINLINE_FUNCTION
    double compute_speed_of_sound(double const density, double const pressure) const noexcept
    {
        return std::sqrt(m_gamma * pressure / density);
    }
};

} // namespace thermodynamics
