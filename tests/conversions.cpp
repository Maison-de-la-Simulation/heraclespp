#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <PerfectGas.hpp>
#include <hydro/array_conversion.hpp>
#include <hydro/euler_equations.hpp>
#include <kokkos_shortcut.hpp>
#include <mesh/range.hpp>

TEST(Conversions, PrimToCons)
{
    int const n = 10;
    novapp::Range const range({0, 0, 0}, {n, n, n}, 0);
    novapp::EOS const eos(2, 1);
    novapp::EulerPrim prim;
    prim.rho = 2;
    for (int idim = 0; idim < novapp::ndim; ++idim)
    {
        prim.u[idim] = -1;
    }
    prim.P = 10;
    novapp::KV_double_3d rho_view("rho", n, n, n);
    novapp::KV_double_4d u_view("u", n, n, n, novapp::ndim);
    novapp::KV_double_3d P_view("P", n, n, n);

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
            rhou_view.view_device(),
            E_view.view_device(),
            rho_view,
            u_view,
            P_view,
            eos);
    rhou_view.modify_device();
    E_view.modify_device();
    rhou_view.sync_host();
    E_view.sync_host();

    novapp::KDV_double_4d::t_host rhou_host = rhou_view.view_host();
    novapp::KDV_double_3d::t_host E_host = E_view.view_host();
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
    novapp::EOS const eos(2, 1);
    novapp::EulerCons cons;
    cons.rho = 2;
    for (int idim = 0; idim < novapp::ndim; ++idim)
    {
        cons.rhou[idim] = -2;
    }
    cons.E = 10;
    novapp::KV_double_3d rho_view("rho", n, n, n);
    novapp::KV_double_4d rhou_view("rhou", n, n, n, novapp::ndim);
    novapp::KV_double_3d E_view("E", n, n, n);

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
            u_view.view_device(),
            P_view.view_device(),
            rho_view,
            rhou_view,
            E_view,
            eos);
    u_view.modify_device();
    P_view.modify_device();
    u_view.sync_host();
    P_view.sync_host();

    novapp::KDV_double_4d::t_host u_host = u_view.view_host();
    novapp::KDV_double_3d::t_host P_host = P_view.view_host();
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
