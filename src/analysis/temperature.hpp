//!
//! @file temperature.hpp
//!

#pragma once

#include <kokkos_shortcut.hpp>
#include <range.hpp>

namespace novapp
{

template <class EoS>
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