#include <cmath>

#include <gtest/gtest.h>

#include <godunov_scheme.hpp>

TEST(RiemannSolver, Consistency)
{
    HLL const riemann_solver;
    thermodynamics::PerfectGas const eos(1.4, 1.);
    EulerPrim prim;
    prim.density = 2.;
    prim.velocity = 3.;
    prim.pressure = 10.;
    EulerCons const cons = to_cons(prim, eos);
    EulerFlux const numerical_flux = riemann_solver(cons, cons, eos);
    EulerFlux const physical_flux = compute_flux(cons, eos);
    EXPECT_DOUBLE_EQ(numerical_flux.density, physical_flux.density);
    EXPECT_DOUBLE_EQ(numerical_flux.momentum, physical_flux.momentum);
    EXPECT_DOUBLE_EQ(numerical_flux.energy, physical_flux.energy);
}

TEST(RiemannSolver, Symmetry)
{
    HLL const riemann_solver;
    thermodynamics::PerfectGas const eos(1.4, 1.);
    EulerPrim prim_left;
    prim_left.density = 2.;
    prim_left.velocity = 3.;
    prim_left.pressure = 10.;
    EulerPrim prim_right;
    prim_right.density = 3.;
    prim_right.velocity = -6.;
    prim_right.pressure = 5.5;
    EulerCons cons_left = to_cons(prim_left, eos);
    EulerCons cons_right = to_cons(prim_right, eos);
    EulerFlux const flux1 = riemann_solver(cons_left, cons_right, eos);
    std::swap(cons_left, cons_right);
    cons_left.momentum = -cons_left.momentum;
    cons_right.momentum = -cons_right.momentum;
    EulerFlux const flux2 = riemann_solver(cons_left, cons_right, eos);
    EXPECT_DOUBLE_EQ(flux1.density, -flux2.density);
    EXPECT_DOUBLE_EQ(flux1.momentum, flux2.momentum);
    EXPECT_DOUBLE_EQ(flux1.energy, -flux2.energy);
}
