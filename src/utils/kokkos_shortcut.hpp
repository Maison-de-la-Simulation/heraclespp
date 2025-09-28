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
#include <Kokkos_DualView.hpp>

namespace novapp
{

using Kokkos::ALL;

using KV_double_1d=Kokkos::View<double*, Kokkos::LayoutLeft>;
using KV_double_2d=Kokkos::View<double**, Kokkos::LayoutLeft>;
using KV_double_3d=Kokkos::View<double***, Kokkos::LayoutLeft>;
using KV_double_4d=Kokkos::View<double****, Kokkos::LayoutLeft>;
using KV_double_5d=Kokkos::View<double*****, Kokkos::LayoutLeft>;
using KV_double_6d=Kokkos::View<double******, Kokkos::LayoutLeft>;

using KVH_double_1d=KV_double_1d::HostMirror;
using KVH_double_2d=KV_double_2d::HostMirror;
using KVH_double_3d=KV_double_3d::HostMirror;
using KVH_double_4d=KV_double_4d::HostMirror;
using KVH_double_5d=KV_double_5d::HostMirror;
using KVH_double_6d=KV_double_6d::HostMirror;

using KV_cdouble_1d=Kokkos::View<const double*, Kokkos::LayoutLeft>;
using KV_cdouble_2d=Kokkos::View<const double**, Kokkos::LayoutLeft>;
using KV_cdouble_3d=Kokkos::View<const double***, Kokkos::LayoutLeft>;
using KV_cdouble_4d=Kokkos::View<const double****, Kokkos::LayoutLeft>;
using KV_cdouble_5d=Kokkos::View<const double*****, Kokkos::LayoutLeft>;
using KV_cdouble_6d=Kokkos::View<const double******, Kokkos::LayoutLeft>;

using KDV_double_1d=Kokkos::DualView<double*, Kokkos::LayoutLeft>;
using KDV_double_2d=Kokkos::DualView<double**, Kokkos::LayoutLeft>;
using KDV_double_3d=Kokkos::DualView<double***, Kokkos::LayoutLeft>;
using KDV_double_4d=Kokkos::DualView<double****, Kokkos::LayoutLeft>;

using KDV_cdouble_3d=Kokkos::DualView<const double***, Kokkos::LayoutLeft>;
using KDV_cdouble_4d=Kokkos::DualView<const double****, Kokkos::LayoutLeft>;

template <class... DualViews>
    requires((Kokkos::is_dual_view_v<DualViews> && ...))
void modify_host(DualViews&... views)
{
    (views.modify_host(), ...);
}

template <class... DualViews>
    requires((Kokkos::is_dual_view_v<DualViews> && ...))
void modify_device(DualViews&... views)
{
    (views.modify_device(), ...);
}

template <class... DualViews>
    requires((Kokkos::is_dual_view_v<DualViews> && ...))
void sync_host(DualViews&... views)
{
    (views.sync_host(), ...);
}

template <class... DualViews>
    requires((Kokkos::is_dual_view_v<DualViews> && ...))
void sync_device(DualViews&... views)
{
    (views.sync_device(), ...);
}

template <class View, std::size_t N>
    requires(Kokkos::is_view_v<View> || Kokkos::is_dual_view_v<View>)
bool equal_extents(std::size_t const i, Kokkos::Array<View, N> const& views) noexcept
{
    return std::ranges::all_of(views, [&](View const& view) -> bool {
        return views[0].extent(i) == view.extent(i);
    });
}

template <class View0, class... Views>
    requires(
            (Kokkos::is_view_v<View0> || Kokkos::is_dual_view_v<View0>)
            && ((Kokkos::is_view_v<Views> || Kokkos::is_dual_view_v<Views>) && ...))
bool equal_extents(std::size_t const i, View0 const& view0, Views const&... views) noexcept
{
    return ((view0.extent(i) == views.extent(i)) && ...);
}

template <class View, std::size_t N>
    requires(Kokkos::is_view_v<View> || Kokkos::is_dual_view_v<View>)
bool equal_extents(
        std::initializer_list<std::size_t> idx,
        Kokkos::Array<View, N> const& views) noexcept
{
    return std::ranges::all_of(idx, [&](std::size_t const i) -> bool {
        return equal_extents(i, views);
    });
}

template <class View0, class... Views>
    requires(
            (Kokkos::is_view_v<View0> || Kokkos::is_dual_view_v<View0>)
            && ((Kokkos::is_view_v<Views> || Kokkos::is_dual_view_v<Views>) && ...))
bool equal_extents(
        std::initializer_list<std::size_t> idx,
        View0 const& view0,
        Views const&... views) noexcept
{
    return std::ranges::all_of(idx, [&](std::size_t const i) -> bool {
        return equal_extents(i, view0, views...);
    });
}

} // namespace novapp
