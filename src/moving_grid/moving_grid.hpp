// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <kokkos_shortcut.hpp>

namespace novapp {

class Grid;

void shift_array(KV_cdouble_3d const& var_old, KV_double_3d const& var_new);

void shift_array(KV_cdouble_4d const& var_old, KV_double_4d const& var_new);

void shift_grid(
        KV_cdouble_3d const& rho_old,
        KV_cdouble_4d const& rhou_old,
        KV_cdouble_3d const& E_old,
        KV_cdouble_4d const& fx_old,
        KV_double_3d const& rho_new,
        KV_double_4d const& rhou_new,
        KV_double_3d const& E_new,
        KV_double_4d const& fx_new,
        KDV_double_1d& x_glob,
        KDV_double_1d& y_glob,
        KDV_double_1d& z_glob,
        Grid& grid);

} // namespace novapp
