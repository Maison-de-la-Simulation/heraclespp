#include <gtest/gtest.h>

#include <PerfectGas.hpp>
#include <godunov_scheme.hpp>

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

using RiemannSolvers = ::testing::Types<novapp::HLL, novapp::HLLC, novapp::Splitting>;
// Trailing comma is needed to avoid spurious `gnu-zero-variadic-macro-arguments` warning with clang
TYPED_TEST_SUITE(RiemannSolverFixture, RiemannSolvers, );

TYPED_TEST(RiemannSolverFixture, Consistency)
{
    novapp::thermodynamics::PerfectGas const eos(1.4, 1.);
    novapp::EulerPrim prim;
    prim.rho = 2.;
    prim.u = {3.};
    prim.P = 10.;
    novapp::EulerCons const cons = to_cons(prim, eos);
    novapp::EulerFlux const numerical_flux = this->m_riemann_solver(cons, cons, 0, eos);
    novapp::EulerFlux const physical_flux = compute_flux(cons, 0, eos);
    EXPECT_DOUBLE_EQ(numerical_flux.rho, physical_flux.rho);
    EXPECT_DOUBLE_EQ(numerical_flux.rhou[0], physical_flux.rhou[0]);
    EXPECT_DOUBLE_EQ(numerical_flux.E, physical_flux.E);
}

TYPED_TEST(RiemannSolverFixture, Symmetry)
{
    novapp::thermodynamics::PerfectGas const eos(1.4, 1.);
    novapp::EulerPrim prim_left;
    prim_left.rho = 2.;
    prim_left.u = {3.};
    prim_left.P = 10.;
    novapp::EulerPrim prim_right;
    prim_right.rho = 3.;
    prim_right.u = {-6.};
    prim_right.P = 5.5;
    novapp::EulerCons cons_left = to_cons(prim_left, eos);
    novapp::EulerCons cons_right = to_cons(prim_right, eos);
    novapp::EulerFlux const flux1 = this->m_riemann_solver(cons_left, cons_right, 0, eos);
    std::swap(cons_left, cons_right);
    cons_left.rhou[0] = -cons_left.rhou[0];
    cons_right.rhou[0] = -cons_right.rhou[0];
    novapp::EulerFlux const flux2 = this->m_riemann_solver(cons_left, cons_right, 0, eos);
    EXPECT_DOUBLE_EQ(flux1.rho, -flux2.rho);
    EXPECT_DOUBLE_EQ(flux1.rhou[0], flux2.rhou[0]);
    EXPECT_DOUBLE_EQ(flux1.E, -flux2.E);
}
