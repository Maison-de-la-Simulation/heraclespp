#pragma once

#include <kokkos_shortcut.hpp>

namespace novapp {

class Range;

void broadcast(Range const& range, double in, KV_double_3d out);

void broadcast(Range const& range, KV_cdouble_1d in, KV_double_3d out);

} // namespace novapp
