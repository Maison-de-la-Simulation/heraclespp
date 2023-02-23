#include <gtest/gtest.h>

#include <Kokkos_DualView.hpp>
#include <Kokkos_Core.hpp>
#include <PerfectGas.hpp>
#include <array_conversion.hpp>
#include <euler_equations.hpp>

TEST(Conversions, PrimToCons)
{
    int const n = 10;
    thermodynamics::PerfectGas const eos(2, 1);
    EulerPrim prim;
    prim.density = 2;
    for (int idim = 0; idim < ndim; ++idim)
    {
        prim.velocity[idim] = -1;
    }
    prim.pressure = 10;
    Kokkos::View<double***> rho_view("rho", n, n, n);
    Kokkos::View<double****> u_view("u", n, n, n, ndim);
    Kokkos::View<double***> P_view("P", n, n, n);

    Kokkos::deep_copy(rho_view, prim.density);
    for (int idim = 0; idim < ndim; ++idim)
    {
        Kokkos::deep_copy(u_view, prim.velocity[idim]);
    }
    Kokkos::deep_copy(P_view, prim.pressure);

    Kokkos::DualView<double****> rhou_view("rhou", n, n, n, ndim);
    Kokkos::DualView<double***> E_view("E", n, n, n);
    ConvPrimtoConsArray(rhou_view.view_device(), E_view.view_device(), rho_view, u_view, P_view, eos);
    rhou_view.modify_device();
    E_view.modify_device();
    rhou_view.sync_host();
    E_view.sync_host();

    Kokkos::DualView<double****>::t_host rhou_host = rhou_view.view_host();
    Kokkos::DualView<double***>::t_host E_host = E_view.view_host();
    EulerCons const cons = to_cons(prim, eos);
    for (int k = 0; k < n; ++k)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int i = 0; i < n; ++i)
            {
                for ( int idim = 0; idim < ndim; ++idim )
                {
                    EXPECT_DOUBLE_EQ(rhou_host(i, j, k, idim), cons.momentum[idim]);
                }
                EXPECT_DOUBLE_EQ(E_host(i, j, k), cons.energy);
            }
        }
    }
}

TEST(Conversions, ConsToPrim)
{
    int const n = 10;
    thermodynamics::PerfectGas const eos(2, 1);
    EulerCons cons;
    cons.density = 2;
    for (int idim = 0; idim < ndim; ++idim)
    {
        cons.momentum[idim] = -2;
    }
    cons.energy = 10;
    Kokkos::View<double***> rho_view("rho", n, n, n);
    Kokkos::View<double****> rhou_view("rhou", n, n, n, ndim);
    Kokkos::View<double***> E_view("E", n, n, n);

    Kokkos::deep_copy(rho_view, cons.density);
    for (int idim = 0; idim < ndim; ++idim)
    {
        Kokkos::deep_copy(rhou_view, cons.momentum[idim]);
    }
    Kokkos::deep_copy(E_view, cons.energy);

    Kokkos::DualView<double****> u_view("u", n, n, n, ndim);
    Kokkos::DualView<double***> P_view("P", n, n, n);
    ConvConstoPrimArray(u_view.view_device(), P_view.view_device(), rho_view, rhou_view, E_view, eos);
    u_view.modify_device();
    P_view.modify_device();
    u_view.sync_host();
    P_view.sync_host();

    Kokkos::DualView<double****>::t_host u_host = u_view.view_host();
    Kokkos::DualView<double***>::t_host P_host = P_view.view_host();
    EulerPrim const prim = to_prim(cons, eos);
    for (int k = 0; k < n; ++k)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int i = 0; i < n; ++i)
            {
                for (int idim = 0; idim < ndim; ++idim)
                {
                    EXPECT_DOUBLE_EQ(u_host(i, j, k, idim), prim.velocity[idim]);
                }
                EXPECT_DOUBLE_EQ(P_host(i, j, k), prim.pressure);
            }
        }
    }
}
