// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <array>
#include <cassert>
#include <string>

#include <Kokkos_Core.hpp>
#include <kokkos_shortcut.hpp>
#include <range.hpp>

#include "broadcast.hpp"

namespace hclpp {

void broadcast(Range const& range, double const in, KV_double_3d const& out)
{
    assert(range.NgEff == 0);
    Kokkos::parallel_for("broadcast 0d->3d", cell_mdrange(range), KOKKOS_LAMBDA(int const i, int const j, int const k) { out(i, j, k) = in; });
}

void broadcast(Range const& range, double const in, KV_double_4d const& out)
{
    for (int i = 0; i < out.extent_int(3); ++i) {
        broadcast(range, in, Kokkos::subview(out, ALL, ALL, ALL, i));
    }
}

void broadcast(Range const& range, KV_cdouble_1d const& in, KV_double_3d const& out)
{
    assert(range.NgEff == 0);
    assert(in.extent_int(0) + (2 * range.Nghost[0] == out.extent_int(0)));
    int const ghost_x = range.Nghost[0];

    Kokkos::parallel_for(
            "broadcast 1d->3d",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int const i, int const j, int const k) { out(i, j, k) = in(i - ghost_x); });
}

void broadcast(Range const& range, KV_cdouble_1d const& in, KV_double_4d const& out)
{
    for (int i = 0; i < out.extent_int(3); ++i) {
        broadcast(range, in, Kokkos::subview(out, ALL, ALL, ALL, i));
    }
}

} // namespace hclpp
