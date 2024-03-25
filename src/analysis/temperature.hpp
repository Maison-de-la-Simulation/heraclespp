//!
//! @file temperature.hpp
//!

#pragma once

#include <eos.hpp>
#include <kokkos_shortcut.hpp>

namespace novapp
{

class Range;

void temperature(
    Range const& range,
    EOS const& eos,
    KV_cdouble_3d rho,
    KV_cdouble_3d P,
    KV_double_3d T);

} // namespace novapp