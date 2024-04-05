//!
//! @file time_step.hpp
//! CFL condition
//!

#pragma once

#include <eos.hpp>
#include <kokkos_shortcut.hpp>

namespace novapp
{

namespace thermodynamics
{

class PerfectGas;
class RadGas;

} // namespace thermodynamics

class Range;
class Grid;

//! Time step with the cfl condition
//! @param[in] range output iteration range
//! @param[in] eos equation of state
//! @param[in] grid mesh metadata
//! @param[in] cfl CFL stability factor to apply
//! @param[in] rho density array 3D
//! @param[in] u velocity array 3D
//! @param[in] P pressure array 3D
//! @return time step
[[nodiscard]] double time_step(
    Range const& range,
    EOS const& eos,
    Grid const& grid,
    double cfl,
    KV_cdouble_3d rho,
    KV_cdouble_4d u,
    KV_cdouble_3d P);

} // namespace novapp
