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

    //! @param[in] Uim1 float U_{i-1}^{n}
    //! @param[in] Ui float U_{i}^{n}
    //! @param[in] Uip1 float U_{i+1}^{n}
    //! @return slope
    double operator()(
            [[maybe_unused]] double const Uim1,
            [[maybe_unused]] double const Ui,
            [[maybe_unused]] double const Uip1) const
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
    //! @param[in] Uim1 float U_{i-1}^{n}
    //! @param[in] Ui float U_{i}^{n}
    //! @param[in] Uip1 float U_{i+1}^{n}
    //! @return slope
    double operator()(double const Uim1, double const Ui, double const Uip1) const
    {
        double const diffR = Uip1 - Ui; // Right slope
        double const diffL = Ui - Uim1; // Left
        double const R = diffR / diffL;
        if (diffL * diffR > 0)
        {
            return (1. / 2) * (diffR + diffL) * (4 * R) / ((R + 1) * (R + 1));
        }
        else
        {
            return 0;
        }
    }
};
