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
    KV_cdouble_3d const& rho,
    KV_cdouble_4d const& rhou,
    KV_cdouble_3d const& E);

} // namespace novapp