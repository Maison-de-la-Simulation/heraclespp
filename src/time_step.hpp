//!
//! @file cfl_cond.hpp
//! CFL condition
//!

#pragma once

#include <mpi.h>

#include "euler_equations.hpp"
#include "range.hpp"
#include "grid.hpp"
#include "kokkos_shortcut.hpp"
#include "eos.hpp"

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
//! @param[in] rho density array 3D
//! @param[in] u celerity array 3D
//! @param[in] P pressure array 3D
//! @return time step
[[nodiscard]] double time_step(
    Range const& range,
    double const cfl,
    KV_cdouble_3d rho,
    KV_cdouble_4d u,
    KV_cdouble_3d P,
    EOS const& eos,
    Grid const& grid);

} // namespace novapp
