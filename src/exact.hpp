/**
 * @file exact.hpp
 * Exact solution for tube shock tube
 */
#pragma once

#include <Kokkos_Core.hpp>

//! Sound speed
//! @param[in] rhok density with k = left or right
//! @param[in] Pk pressure with k = left or right
//! @return sound speed
double sound_speed(
        double rhok,
        double Pk);
