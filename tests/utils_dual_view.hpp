// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_DualView.hpp>

namespace hclpp {

template <class DualView>
    requires(Kokkos::is_dual_view_v<DualView>)
auto view_device(DualView const& view) noexcept
{
    return view.view_device();
}

template <class DualView>
    requires(Kokkos::is_dual_view_v<DualView>)
auto view_host(DualView const& view) noexcept
{
    return view.view_host();
}

} // namespace hclpp
