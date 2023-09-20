//!
//! @file kokkos_shortcut.hpp
//!

#pragma once

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
void modify_host(DualViews&... views)
{
    (views.modify_host(),...);
}

template <class... DualViews>
void modify_device(DualViews&... views)
{
    (views.modify_device(),...);
}

template <class... DualViews>
void sync_host(DualViews&... views)
{
    (views.sync_host(),...);
}

template <class... DualViews>
void sync_device(DualViews&... views)
{
    (views.sync_device(),...);
}

} // namespace novapp
