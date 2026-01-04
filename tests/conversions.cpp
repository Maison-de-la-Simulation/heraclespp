// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <string>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <array_conversion.hpp>
#include <euler_equations.hpp>
#include <kokkos_shortcut.hpp>
#include <ndim.hpp>
#include <perfect_gas.hpp>
#include <range.hpp>

TEST(Conversions, PrimToCons)
{
    int const n = 10;
    double const rho0 = 2.;
    double const u0 = -1.;
    double const P0 = 10.;

    hclpp::Range const range({0, 0, 0}, {n, n, n}, 0);
    hclpp::thermodynamics::PerfectGas const eos(2, 1);
    hclpp::EulerPrim prim;
    prim.rho = rho0;
    for (int idim = 0; idim < hclpp::ndim; ++idim) {
        prim.u[idim] = u0;
    }
    prim.P = P0;
    hclpp::KV_double_3d const rho_view("rho", n, n, n);
    hclpp::KV_double_4d const u_view("u", n, n, n, hclpp::ndim);
    hclpp::KV_double_3d const P_view("P", n, n, n);

    Kokkos::deep_copy(rho_view, prim.rho);
    for (int idim = 0; idim < hclpp::ndim; ++idim) {
        Kokkos::deep_copy(u_view, prim.u[idim]);
    }
    Kokkos::deep_copy(P_view, prim.P);

    hclpp::KDV_double_4d const rhou_view("rhou", n, n, n, hclpp::ndim);
    hclpp::KDV_double_3d const E_view("E", n, n, n);
    conv_prim_to_cons(range.all_ghosts(), eos, rho_view, u_view, P_view, rhou_view(hclpp::device), E_view(hclpp::device));

    auto const rhou_host = rhou_view(hclpp::host, hclpp::read_only);
    auto const E_host = E_view(hclpp::host, hclpp::read_only);
    hclpp::EulerCons const cons = to_cons(prim, eos);
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                for (int idim = 0; idim < hclpp::ndim; ++idim) {
                    EXPECT_DOUBLE_EQ(rhou_host(i, j, k, idim), cons.rhou[idim]);
                }
                EXPECT_DOUBLE_EQ(E_host(i, j, k), cons.E);
            }
        }
    }
}

TEST(Conversions, ConsToPrim)
{
    int const n = 10;
    double const rho0 = 2.;
    double const rhou0 = -2.;
    double const E0 = 10.;

    hclpp::Range const range({0, 0, 0}, {n, n, n}, 0);
    hclpp::thermodynamics::PerfectGas const eos(2, 1);
    hclpp::EulerCons cons;
    cons.rho = rho0;
    for (int idim = 0; idim < hclpp::ndim; ++idim) {
        cons.rhou[idim] = rhou0;
    }
    cons.E = E0;
    hclpp::KV_double_3d const rho_view("rho", n, n, n);
    hclpp::KV_double_4d const rhou_view("rhou", n, n, n, hclpp::ndim);
    hclpp::KV_double_3d const E_view("E", n, n, n);

    Kokkos::deep_copy(rho_view, cons.rho);
    for (int idim = 0; idim < hclpp::ndim; ++idim) {
        Kokkos::deep_copy(rhou_view, cons.rhou[idim]);
    }
    Kokkos::deep_copy(E_view, cons.E);

    hclpp::KDV_double_4d const u_view("u", n, n, n, hclpp::ndim);
    hclpp::KDV_double_3d const P_view("P", n, n, n);
    conv_cons_to_prim(range.all_ghosts(), eos, rho_view, rhou_view, E_view, u_view(hclpp::device), P_view(hclpp::device));

    auto const u_host = u_view(hclpp::host, hclpp::read_only);
    auto const P_host = P_view(hclpp::host, hclpp::read_only);
    hclpp::EulerPrim const prim = to_prim(cons, eos);
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                for (int idim = 0; idim < hclpp::ndim; ++idim) {
                    EXPECT_DOUBLE_EQ(u_host(i, j, k, idim), prim.u[idim]);
                }
                EXPECT_DOUBLE_EQ(P_host(i, j, k), prim.P);
            }
        }
    }
}
