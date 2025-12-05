// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <utility>

#include <gtest/gtest.h>

#include <Kokkos_Array.hpp>
#include <euler_equations.hpp>
#include <perfect_gas.hpp>
#include <riemann_solver.hpp>

template <class RiemannSolver>
class RiemannSolverFixture : public ::testing::Test
{
public:
    RiemannSolver m_riemann_solver;

    RiemannSolverFixture() = default;

    RiemannSolverFixture(RiemannSolverFixture const& rhs) = default;

    RiemannSolverFixture(RiemannSolverFixture&& rhs) noexcept = default;

    ~RiemannSolverFixture() override = default;

    RiemannSolverFixture& operator=(RiemannSolverFixture const& rhs) = default;

    RiemannSolverFixture& operator=(RiemannSolverFixture&& rhs) noexcept = default;
};

using RiemannSolvers = ::testing::Types<hclpp::HLL, hclpp::HLLC, hclpp::Splitting>;
// Trailing comma is needed to avoid spurious `gnu-zero-variadic-macro-arguments` warning with clang
TYPED_TEST_SUITE(RiemannSolverFixture, RiemannSolvers, );

TYPED_TEST(RiemannSolverFixture, Consistency)
{
    double const gamma = 1.4;
    double const mmw = 1.;
    double const rho0 = 2.;
    double const u0 = 3.;
    double const P0 = 10.;

    hclpp::thermodynamics::PerfectGas const eos(gamma, mmw);
    hclpp::EulerPrim const prim {.rho = rho0, .u = {u0}, .P = P0};
    hclpp::EulerCons const cons = to_cons(prim, eos);
    hclpp::EulerFlux const numerical_flux = this->m_riemann_solver(cons, cons, 0, eos);
    hclpp::EulerFlux const physical_flux = compute_flux(cons, 0, eos);
    EXPECT_DOUBLE_EQ(numerical_flux.rho, physical_flux.rho);
    EXPECT_DOUBLE_EQ(numerical_flux.rhou[0], physical_flux.rhou[0]);
    EXPECT_DOUBLE_EQ(numerical_flux.E, physical_flux.E);
}

TYPED_TEST(RiemannSolverFixture, Symmetry)
{
    double const gamma = 1.4;
    double const mmw = 1.;
    double const rho_left = 2.;
    double const u_left = 3.;
    double const P_left = 10.;
    double const rho_right = 3.;
    double const u_right = -6.;
    double const P_right = 5.5;

    hclpp::thermodynamics::PerfectGas const eos(gamma, mmw);
    hclpp::EulerPrim const prim_left {.rho = rho_left, .u = {u_left}, .P = P_left};
    hclpp::EulerPrim const prim_right {.rho = rho_right, .u = {u_right}, .P = P_right};
    hclpp::EulerCons cons_left = to_cons(prim_left, eos);
    hclpp::EulerCons cons_right = to_cons(prim_right, eos);
    hclpp::EulerFlux const flux1 = this->m_riemann_solver(cons_left, cons_right, 0, eos);
    std::swap(cons_left, cons_right);
    cons_left.rhou[0] = -cons_left.rhou[0];
    cons_right.rhou[0] = -cons_right.rhou[0];
    hclpp::EulerFlux const flux2 = this->m_riemann_solver(cons_left, cons_right, 0, eos);
    EXPECT_DOUBLE_EQ(flux1.rho, -flux2.rho);
    EXPECT_DOUBLE_EQ(flux1.rhou[0], flux2.rhou[0]);
    EXPECT_DOUBLE_EQ(flux1.E, -flux2.E);
}
