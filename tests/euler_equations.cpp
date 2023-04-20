#include <gtest/gtest.h>

#include <euler_equations.hpp>

TEST(EulerEquations, KineticEnergyFromPrim)
{
    novapp::EulerPrim prim;
    prim.rho = 2.;
    prim.u = {10.};
    prim.P = 0.;
    EXPECT_DOUBLE_EQ(compute_ek(prim), 100.);
}

TEST(EulerEquations, KineticEnergyFromCons)
{
    novapp::EulerCons cons;
    cons.rho = 2.;
    cons.rhou = {20.};
    cons.E = 100.;
    EXPECT_DOUBLE_EQ(compute_ek(cons), 100.);
}

TEST(EulerEquations, PrimToConsToPrim)
{
    novapp::thermodynamics::PerfectGas eos(1.4, 1.);
    novapp::EulerPrim prim;
    prim.rho = 2.;
    prim.u = {10.};
    prim.P = 1.;
    novapp::EulerPrim prim_to_cons_to_prim = to_prim(to_cons(prim, eos), eos);
    EXPECT_DOUBLE_EQ(prim.rho, prim_to_cons_to_prim.rho);
    EXPECT_DOUBLE_EQ(prim.P, prim_to_cons_to_prim.P);
    EXPECT_DOUBLE_EQ(prim.u[0], prim_to_cons_to_prim.u[0]);
}
