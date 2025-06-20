// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file temperature.hpp
//!

#pragma once

#include <Kokkos_Core.hpp>
#include <concepts.hpp>
#include <kokkos_shortcut.hpp>
#include <range.hpp>

namespace novapp
{

template <concepts::EulerEoS EoS>
void temperature(
    Range const& range,
    EoS const& eos,
    KV_cdouble_3d const& rho,
    KV_cdouble_3d const& P,
    KV_double_3d const& T)
{
    Kokkos::parallel_for(
        "fill_temperature_array",
        cell_mdrange(range),
        KOKKOS_LAMBDA(int i, int j, int k)
        {
            T(i, j, k) = eos.compute_T_from_P(rho(i, j, k), P(i, j, k));
        });
}

} // namespace novapp
