#include <gtest/gtest.h>

#include <euler_equations.hpp>

TEST(EulerEquations, KineticEnergyFromPrim)
{
    EulerPrim prim;
    prim.density = 2.;
    prim.velocity = {10.};
    prim.pressure = 0.;
    EXPECT_DOUBLE_EQ(compute_volumic_kinetic_energy(prim), 100.);
}

TEST(EulerEquations, KineticEnergyFromCons)
{
    EulerCons cons;
    cons.density = 2.;
    cons.momentum = {20.};
    cons.energy = 100.;
    EXPECT_DOUBLE_EQ(compute_volumic_kinetic_energy(cons), 100.);
}

TEST(EulerEquations, PrimToConsToPrim)
{
    thermodynamics::PerfectGas eos(1.4, 1.);
    EulerPrim prim;
    prim.density = 2.;
    prim.velocity = {10.};
    prim.pressure = 1.;
    EulerPrim prim_to_cons_to_prim = to_prim(to_cons(prim, eos), eos);
    EXPECT_DOUBLE_EQ(prim.density, prim_to_cons_to_prim.density);
    EXPECT_DOUBLE_EQ(prim.pressure, prim_to_cons_to_prim.pressure);
    EXPECT_DOUBLE_EQ(prim.velocity[0], prim_to_cons_to_prim.velocity[0]);
}
