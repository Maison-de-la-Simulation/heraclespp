// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file boundary.cpp
//!

#include <array>
#include <string>
#include <string_view>

#include <Kokkos_Core.hpp>
#include <grid.hpp>
#include <kokkos_shortcut.hpp>

#include "boundary.hpp"

namespace hclpp {

std::string_view bc_dir(int i)
{
    static constexpr std::array<std::string_view, 3> s_bc_dir {"_X0", "_X1", "_X2"};
    return s_bc_dir.at(i);
}

std::string_view bc_face(int i)
{
    static constexpr std::array<std::string_view, 2> s_bc_face {"_left", "_right"};
    return s_bc_face.at(i);
}

void null_gradient_condition(
        int bc_idim,
        int bc_iface,
        std::string const& label,
        Grid const& grid,
        KV_double_3d const& rho,
        KV_double_4d const& rhou,
        KV_double_3d const& E,
        KV_double_4d const& fx)
{
    Kokkos::Array<int, 3> begin {0, 0, 0};
    Kokkos::Array<int, 3> end {rho.extent_int(0), rho.extent_int(1), rho.extent_int(2)};
    int const nfx = fx.extent_int(3);

    int const ng = grid.Nghost[bc_idim];
    if (bc_iface == 1) {
        begin[bc_idim] = rho.extent_int(bc_idim) - ng;
    }
    end[bc_idim] = begin[bc_idim] + ng;

    int const offset = bc_iface == 0 ? end[bc_idim] : begin[bc_idim] - 1;
    Kokkos::parallel_for(
            label,
            Kokkos::MDRangePolicy<Kokkos::IndexType<int>, Kokkos::Rank<3>>(begin, end),
            KOKKOS_LAMBDA(int i, int j, int k) {
                Kokkos::Array<int, 3> offsets {i, j, k};
                offsets[bc_idim] = offset;
                rho(i, j, k) = rho(offsets[0], offsets[1], offsets[2]);
                for (int n = 0; n < rhou.extent_int(3); ++n) {
                    rhou(i, j, k, n) = rhou(offsets[0], offsets[1], offsets[2], n);
                }
                E(i, j, k) = E(offsets[0], offsets[1], offsets[2]);
                for (int ifx = 0; ifx < nfx; ++ifx) {
                    fx(i, j, k, ifx) = fx(offsets[0], offsets[1], offsets[2], ifx);
                }
            });
}

void reflexive_condition(
        int bc_idim,
        int bc_iface,
        std::string const& label,
        Grid const& grid,
        KV_double_3d const& rho,
        KV_double_4d const& rhou,
        KV_double_3d const& E,
        KV_double_4d const& fx)
{
    Kokkos::Array<int, 3> begin {0, 0, 0};
    Kokkos::Array<int, 3> end {rho.extent_int(0), rho.extent_int(1), rho.extent_int(2)};
    int const nfx = fx.extent_int(3);

    int const ng = grid.Nghost[bc_idim];
    if (bc_iface == 1) {
        begin[bc_idim] = rho.extent_int(bc_idim) - ng;
    }
    end[bc_idim] = begin[bc_idim] + ng;

    int const mirror = bc_iface == 0 ? ((2 * ng) - 1) : ((2 * (rho.extent_int(bc_idim) - ng)) - 1);
    Kokkos::parallel_for(
            label,
            Kokkos::MDRangePolicy<Kokkos::IndexType<int>, Kokkos::Rank<3>>(begin, end),
            KOKKOS_LAMBDA(int i, int j, int k) {
                Kokkos::Array<int, 3> offsets {i, j, k};
                offsets[bc_idim] = mirror - offsets[bc_idim];
                rho(i, j, k) = rho(offsets[0], offsets[1], offsets[2]);
                for (int n = 0; n < rhou.extent_int(3); ++n) {
                    rhou(i, j, k, n) = rhou(offsets[0], offsets[1], offsets[2], n);
                }
                rhou(i, j, k, bc_idim) = -rhou(i, j, k, bc_idim);
                E(i, j, k) = E(offsets[0], offsets[1], offsets[2]);
                for (int ifx = 0; ifx < nfx; ++ifx) {
                    fx(i, j, k, ifx) = fx(offsets[0], offsets[1], offsets[2], ifx);
                }
            });
}

} // namespace hclpp
