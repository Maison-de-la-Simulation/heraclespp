//!
//! @file slope_limiters.hpp
//! Slope limiters.
//!

#pragma once

#include <Kokkos_Core.hpp>

namespace novapp
{

//! The null slope limiter.
class Constant
{
public:
    //! @param[in] diffR float (U_{i+1}^{n} - U_{i}^{n}) / dx
    //! @param[in] diffL float (U_{i}^{n} - U_{i-1}^{n}) / dx
    //! @return slope
    KOKKOS_FORCEINLINE_FUNCTION
    double operator()([[maybe_unused]] double const diffR, [[maybe_unused]] double const diffL)
            const noexcept
    {
        return 0;
    }
};

//! The Van Leer slope limiter.
class VanLeer
{
public:
    //! The Van Leer formula.
    //! @param[in] diffR float (U_{i+1}^{n} - U_{i}^{n}) / dx
    //! @param[in] diffL float (U_{i}^{n} - U_{i-1}^{n}) / dx
    //! @return slope
    KOKKOS_FORCEINLINE_FUNCTION
    double operator()(double const diffR, double const diffL) const noexcept
    {
        if (diffL * diffR > 0)
        {
            double const ratio = diffR / diffL;
            return (1. / 2) * (diffR + diffL) * (4 * ratio) / ((ratio + 1) * (ratio + 1));
        }

        return 0;
    }
};

//! The Minmod slope limiter.
class Minmod
{
public:
    //! The Minmod formula.
    //! @param[in] diffR float (U_{i+1}^{n} - U_{i}^{n}) / dx
    //! @param[in] diffL float (U_{i}^{n} - U_{i-1}^{n}) / dx
    //! @return slope
    KOKKOS_FORCEINLINE_FUNCTION
    double operator()(double const diffR, double const diffL) const noexcept
    {
        if (diffL * diffR > 0)
        {
            double const ratio = diffR / diffL;
            double const minmod = 2 * Kokkos::fmin(1., ratio) / (1 + ratio);
            return minmod * (diffL + diffR) / 2;
        }

        return 0;
    }
};

//! The Van Albada slope limiter.
class VanAlbada
{
public:
    //! The Van Albada formula.
    //! @param[in] diffR float (U_{i+1}^{n} - U_{i}^{n}) / dx
    //! @param[in] diffL float (U_{i}^{n} - U_{i-1}^{n}) / dx
    //! @return slope
    KOKKOS_FORCEINLINE_FUNCTION
    double operator()(double const diffR, double const diffL) const noexcept
    {
        if (diffL * diffR > 0)
        {
            double const ratio = diffR / diffL;
            return (1. / 2) * (diffR + diffL) * (2 * ratio) / (ratio * ratio + 1);
        }

        return 0;
    }
};

} // namespace novapp
