//!
//! @file internal_energy.hpp
//!

#pragma once

#include <kokkos_shortcut.hpp>

namespace novapp
{

class Range;
class Grid;

[[nodiscard]] double internal_energy(
    Range const& range,
    Grid const& grid,
    KV_cdouble_3d rho,
    KV_cdouble_4d rhou,
    KV_cdouble_3d E);

} // namespace novapp