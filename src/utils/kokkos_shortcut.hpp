// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file kokkos_shortcut.hpp
//!

#pragma once

#include <algorithm>
#include <cstddef>
#include <initializer_list>

#include <Kokkos_Core.hpp>
#include <dual_view.hpp>

namespace hclpp {

using Kokkos::ALL;

using DefaultSpace = Kokkos::DefaultExecutionSpace::memory_space;
using DefaultMemoryTraits = Kokkos::MemoryTraits<>;

using KV_double_1d = Kokkos::View<double*, Kokkos::LayoutLeft, DefaultSpace, DefaultMemoryTraits>;
using KV_double_2d = Kokkos::View<double**, Kokkos::LayoutLeft, DefaultSpace, DefaultMemoryTraits>;
using KV_double_3d = Kokkos::View<double***, Kokkos::LayoutLeft, DefaultSpace, DefaultMemoryTraits>;
using KV_double_4d = Kokkos::View<double****, Kokkos::LayoutLeft, DefaultSpace, DefaultMemoryTraits>;
using KV_double_5d = Kokkos::View<double*****, Kokkos::LayoutLeft, DefaultSpace, DefaultMemoryTraits>;
using KV_double_6d = Kokkos::View<double******, Kokkos::LayoutLeft, DefaultSpace, DefaultMemoryTraits>;

using KVH_double_1d = Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace, DefaultMemoryTraits>;
using KVH_double_2d = Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace, DefaultMemoryTraits>;
using KVH_double_3d = Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::HostSpace, DefaultMemoryTraits>;
using KVH_double_4d = Kokkos::View<double****, Kokkos::LayoutLeft, Kokkos::HostSpace, DefaultMemoryTraits>;
using KVH_double_5d = Kokkos::View<double*****, Kokkos::LayoutLeft, Kokkos::HostSpace, DefaultMemoryTraits>;
using KVH_double_6d = Kokkos::View<double******, Kokkos::LayoutLeft, Kokkos::HostSpace, DefaultMemoryTraits>;

using KV_cdouble_1d = Kokkos::View<double const*, Kokkos::LayoutLeft, DefaultSpace, DefaultMemoryTraits>;
using KV_cdouble_2d = Kokkos::View<double const**, Kokkos::LayoutLeft, DefaultSpace, DefaultMemoryTraits>;
using KV_cdouble_3d = Kokkos::View<double const***, Kokkos::LayoutLeft, DefaultSpace, DefaultMemoryTraits>;
using KV_cdouble_4d = Kokkos::View<double const****, Kokkos::LayoutLeft, DefaultSpace, DefaultMemoryTraits>;
using KV_cdouble_5d = Kokkos::View<double const*****, Kokkos::LayoutLeft, DefaultSpace, DefaultMemoryTraits>;
using KV_cdouble_6d = Kokkos::View<double const******, Kokkos::LayoutLeft, DefaultSpace, DefaultMemoryTraits>;

using KDV_double_1d = DualView<double*, Kokkos::LayoutLeft>;
using KDV_double_2d = DualView<double**, Kokkos::LayoutLeft>;
using KDV_double_3d = DualView<double***, Kokkos::LayoutLeft>;
using KDV_double_4d = DualView<double****, Kokkos::LayoutLeft>;

using KDV_cdouble_1d = DualView<double const*, Kokkos::LayoutLeft>;
using KDV_cdouble_2d = DualView<double const**, Kokkos::LayoutLeft>;
using KDV_cdouble_3d = DualView<double const***, Kokkos::LayoutLeft>;
using KDV_cdouble_4d = DualView<double const****, Kokkos::LayoutLeft>;

template <class View, std::size_t N>
    requires(Kokkos::is_view_v<View>)
auto equal_extents(std::size_t const i, Kokkos::Array<View, N> const& views) noexcept -> bool
{
    return std::ranges::all_of(views, [&](View const& view) -> bool { return views[0].extent(i) == view.extent(i); });
}

template <class View0, class... Views>
    requires(Kokkos::is_view_v<View0> && (Kokkos::is_view_v<Views> && ...))
auto equal_extents(std::size_t const i, View0 const& view0, Views const&... views) noexcept -> bool
{
    return ((view0.extent(i) == views.extent(i)) && ...);
}

template <class View, std::size_t N>
    requires(Kokkos::is_view_v<View>)
auto equal_extents(std::initializer_list<std::size_t> idx, Kokkos::Array<View, N> const& views) noexcept -> bool
{
    return std::ranges::all_of(idx, [&](std::size_t const i) -> bool { return equal_extents(i, views); });
}

template <class View0, class... Views>
    requires(Kokkos::is_view_v<View0> && (Kokkos::is_view_v<Views> && ...))
auto equal_extents(std::initializer_list<std::size_t> idx, View0 const& view0, Views const&... views) noexcept -> bool
{
    return std::ranges::all_of(idx, [&](std::size_t const i) -> bool { return equal_extents(i, view0, views...); });
}

} // namespace hclpp
