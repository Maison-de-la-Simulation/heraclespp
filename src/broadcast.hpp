#pragma once

#include <kokkos_shortcut.hpp>

namespace novapp {

class Grid;
class Range;

void broadcast(Range const& range, Grid const& grid, double in, KV_double_3d out);

void broadcast(Range const& range, Grid const& grid, KV_cdouble_1d in, KV_double_3d out);

} // namespace novapp
