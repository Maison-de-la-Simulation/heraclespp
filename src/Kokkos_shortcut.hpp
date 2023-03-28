//!
//! @file Kokkos_shortcut.hpp
//!

#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

using Kokkos::ALL;

using KV_double_1d=Kokkos::View<double*, Kokkos::LayoutLeft>;
using KV_double_2d=Kokkos::View<double**, Kokkos::LayoutLeft>;
using KV_double_3d=Kokkos::View<double***, Kokkos::LayoutLeft>;
using KV_double_4d=Kokkos::View<double****, Kokkos::LayoutLeft>;
using KV_double_5d=Kokkos::View<double*****, Kokkos::LayoutLeft>;
using KV_double_6d=Kokkos::View<double******, Kokkos::LayoutLeft>;

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


