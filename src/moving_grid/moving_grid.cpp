// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <cassert>

#include <Kokkos_Core.hpp>
#include <grid.hpp>
#include <kokkos_shortcut.hpp>

#include "moving_grid.hpp"

namespace novapp {

void shift_grid(
        KV_cdouble_3d const& rho_old,
        KV_cdouble_4d const& rhou_old,
        KV_cdouble_3d const& E_old,
        KV_cdouble_4d const& fx_old,
        KV_double_3d const& rho_new,
        KV_double_4d const& rhou_new,
        KV_double_3d const& E_new,
        KV_double_4d const& fx_new,
        KDV_double_1d& x0_glob,
        KDV_double_1d& x1_glob,
        KDV_double_1d& x2_glob,
        Grid& grid)
{
    // Shift of physical variables
    shift_array(rho_old, rho_new);
    shift_array(E_old, E_new);
    shift_array(rhou_old, rhou_new);
    shift_array(fx_old, fx_new);

    // Shift of the Grid (only the first dimension)
    {
        x0_glob.sync_host();
        auto const x0_h = x0_glob.view_host();
        int const n = x0_h.extent_int(0);
        for (int i = 0; i < n - 1; ++i) {
            x0_h(i) = x0_h(i + 1);
        }
        // Define the new value for the last interface. We keep the same ratio between the dX (i.e. it will preserve a uniform and a log grid)
        double const r = (x0_h(n - 2) - x0_h(n - 3)) / (x0_h(n - 3) - x0_h(n - 4));
        x0_h(n - 1) = x0_h(n - 2) + ((x0_h(n - 2) - x0_h(n - 3)) * r);
        x0_glob.modify_host();
    }
    x0_glob.sync_device();
    grid.set_grid(x0_glob.view_device(), x1_glob.view_device(), x2_glob.view_device());

    // Boundary conditions are done in the main
}

void shift_array(KV_cdouble_3d const& var_old, KV_double_3d const& var_new)
{
    assert(var_new != var_old);
    assert(var_new.layout() == var_old.layout());

    auto const dst = Kokkos::subview(var_new, Kokkos::make_pair(0, var_new.extent_int(0) - 1), ALL, ALL);
    auto const src = Kokkos::subview(var_old, Kokkos::make_pair(1, var_old.extent_int(0)), ALL, ALL);
    Kokkos::deep_copy(dst, src);
}

void shift_array(KV_cdouble_4d const& var_old, KV_double_4d const& var_new)
{
    assert(var_new != var_old);
    assert(var_new.layout() == var_old.layout());

    for (int i = 0; i < var_old.extent_int(3); ++i) {
        shift_array(
                Kokkos::subview(var_old, ALL, ALL, ALL, i),
                Kokkos::subview(var_new, ALL, ALL, ALL, i));
    }
}

} // namespace novapp
