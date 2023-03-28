//!
//! @file cfl_cond.hpp
//! CFL condition
//!

#pragma once

#include "Kokkos_shortcut.hpp"

namespace novapp
{

namespace thermodynamics
{

class PerfectGas;

} // namespace thermodynamics

class Range;

class Grid;

//! Time step with the cfl condition
//! @param[in] rho density array 3D
//! @param[in] u celerity array 3D
//! @param[in] P pressure array 3D
//! @param[in] dx space step array 3D
//! @return time step
[[nodiscard]] double time_step(
    Range const& range,
    double const cfl,
    KV_cdouble_3d rho,
    KV_cdouble_4d u,
    KV_cdouble_3d P,
    thermodynamics::PerfectGas const& eos,
    Grid const& grid);

} // namespace novapp
