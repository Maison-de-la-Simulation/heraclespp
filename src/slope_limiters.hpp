//!
//! @file slope_limiters.hpp
//! Slope limiters.
//!
#pragma once

#include <string_view>

//! The null slope limiter.
class Null
{
public:
    static constexpr std::string_view s_label = "Null";

    //! @param[in] diffR float (U_{i+1}^{n} - U_{i}^{n}) / dx
    //! @param[in] diffL float (U_{i-1}^{n} - U_{i}^{n}) / dx
    //! @return slope
    double operator()(
            [[maybe_unused]] double const diffR,
            [[maybe_unused]] double const diffL) const
    {
        return 0;
    }
};

//! The Van Leer slope limiter.
class VanLeer
{
public:
    static constexpr std::string_view s_label = "VanLeer";

    //! The Van Leer formula.
    //! @param[in] diffR float (U_{i+1}^{n} - U_{i}^{n}) / dx
    //! @param[in] diffL float (U_{i-1}^{n} - U_{i}^{n}) / dx
    //! @return slope
    double operator()(double const diffR, double const diffL) const
    {
        if (diffL * diffR > 0)
        {
            double const R = diffR / diffL;
            return (1. / 2) * (diffR + diffL) * (4 * R) / ((R + 1) * (R + 1));
        }
        else
        {
            return 0;
        }
    }
};

//! The Minmod slope limiter.
class Minmod
{
public:
    static constexpr std::string_view s_label = "Minmod";
    
    //! The Minmod formula.
    //! @param[in] diffR float (U_{i+1}^{n} - U_{i}^{n}) / dx
    //! @param[in] diffL float (U_{i-1}^{n} - U_{i}^{n}) / dx
    //! @return slope
    double operator()(double const diffR, double const diffL) const
    {
        if (diffL * diffR > 0)
        {
            return 1. / ((1. / diffL) + (1. / diffR));
        }
        else
        {
            return 0;
        }
    }
};

//! The Van Albada slope limiter.
class VanAlbada
{
public:
    static constexpr std::string_view s_label = "VanAlbada";
    
    //! The Minmod formula.
    //! @param[in] diffR float (U_{i+1}^{n} - U_{i}^{n}) / dx
    //! @param[in] diffL float (U_{i-1}^{n} - U_{i}^{n}) / dx
    //! @return slope
    double operator()(double const diffR, double const diffL) const
    {
        
        if (diffL * diffR > 0)
        {
            double const R = diffR / diffL;
            return (1. / 2) * (diffR + diffL) * (2 * R) / (R * R + 1);
        }
        else
        {
            return 0;
        }
    }
};
