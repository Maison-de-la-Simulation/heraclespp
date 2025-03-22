//!
//! @file integration.hpp
//!

#pragma once

#include <kokkos_shortcut.hpp>

namespace novapp {

class Grid;
class Range;

[[nodiscard]] double integrate(Range const& range, Grid const& grid, KV_cdouble_3d const& var);

} // namespace novapp
