/**
 * @file speed_sound.hpp
 * speed of sound
 */

#pragma once

//! Conversion primary to conservative variables
//! @param[inout] rho density value
//! @param[in] P pressure value
//! @param[in] gamma
//! @param[out] c speed sound
inline double speed_sound2(
    double rho,
    double P, 
    double gamma)
{
    return std::sqrt(gamma * P / rho);
};