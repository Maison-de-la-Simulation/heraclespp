// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <string>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <PerfectGas.hpp>
#include <array_conversion.hpp>
#include <euler_equations.hpp>
#include <kokkos_shortcut.hpp>
#include <ndim.hpp>
#include <range.hpp>

TEST(Conversions, PrimToCons)
{
    int const n = 10;
    novapp::Range const range({0, 0, 0}, {n, n, n}, 0);
    novapp::thermodynamics::PerfectGas const eos(2, 1);
    novapp::EulerPrim prim;
    prim.rho = 2;
    for (int idim = 0; idim < novapp::ndim; ++idim)
    {
        prim.u[idim] = -1;
    }
    prim.P = 10;
    novapp::KV_double_3d const rho_view("rho", n, n, n);
    novapp::KV_double_4d const u_view("u", n, n, n, novapp::ndim);
    novapp::KV_double_3d const P_view("P", n, n, n);

    Kokkos::deep_copy(rho_view, prim.rho);
    for (int idim = 0; idim < novapp::ndim; ++idim)
    {
        Kokkos::deep_copy(u_view, prim.u[idim]);
    }
    Kokkos::deep_copy(P_view, prim.P);

    novapp::KDV_double_4d rhou_view("rhou", n, n, n, novapp::ndim);
    novapp::KDV_double_3d E_view("E", n, n, n);
    conv_prim_to_cons(
            range.all_ghosts(),
            eos,
            rho_view,
            u_view,
            P_view,
            rhou_view.view_device(),
            E_view.view_device());
    rhou_view.modify_device();
    E_view.modify_device();
    rhou_view.sync_host();
    E_view.sync_host();

    novapp::KDV_double_4d::t_host const rhou_host = rhou_view.view_host();
    novapp::KDV_double_3d::t_host const E_host = E_view.view_host();
    novapp::EulerCons const cons = to_cons(prim, eos);
    for (int k = 0; k < n; ++k)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int i = 0; i < n; ++i)
            {
                for (int idim = 0; idim < novapp::ndim; ++idim)
                {
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
    novapp::Range const range({0, 0, 0}, {n, n, n}, 0);
    novapp::thermodynamics::PerfectGas const eos(2, 1);
    novapp::EulerCons cons;
    cons.rho = 2;
    for (int idim = 0; idim < novapp::ndim; ++idim)
    {
        cons.rhou[idim] = -2;
    }
    cons.E = 10;
    novapp::KV_double_3d const rho_view("rho", n, n, n);
    novapp::KV_double_4d const rhou_view("rhou", n, n, n, novapp::ndim);
    novapp::KV_double_3d const E_view("E", n, n, n);

    Kokkos::deep_copy(rho_view, cons.rho);
    for (int idim = 0; idim < novapp::ndim; ++idim)
    {
        Kokkos::deep_copy(rhou_view, cons.rhou[idim]);
    }
    Kokkos::deep_copy(E_view, cons.E);

    novapp::KDV_double_4d u_view("u", n, n, n, novapp::ndim);
    novapp::KDV_double_3d P_view("P", n, n, n);
    conv_cons_to_prim(
            range.all_ghosts(),
            eos,
            rho_view,
            rhou_view,
            E_view,
            u_view.view_device(),
            P_view.view_device());
    u_view.modify_device();
    P_view.modify_device();
    u_view.sync_host();
    P_view.sync_host();

    novapp::KDV_double_4d::t_host const u_host = u_view.view_host();
    novapp::KDV_double_3d::t_host const P_host = P_view.view_host();
    novapp::EulerPrim const prim = to_prim(cons, eos);
    for (int k = 0; k < n; ++k)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int i = 0; i < n; ++i)
            {
                for (int idim = 0; idim < novapp::ndim; ++idim)
                {
                    EXPECT_DOUBLE_EQ(u_host(i, j, k, idim), prim.u[idim]);
                }
                EXPECT_DOUBLE_EQ(P_host(i, j, k), prim.P);
            }
        }
    }
}
