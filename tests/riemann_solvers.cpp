#include <gtest/gtest.h>

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

using RiemannSolvers = ::testing::Types<novapp::HLL>;
TYPED_TEST_SUITE(RiemannSolverFixture, RiemannSolvers);

TYPED_TEST(RiemannSolverFixture, Consistency)
{
    novapp::thermodynamics::PerfectGas const eos(1.4, 1., 0);
    novapp::EulerPrim prim;
    prim.density = 2.;
    prim.velocity = {3.};
    prim.pressure = 10.;
    novapp::EulerCons const cons = to_cons(prim, eos);
    novapp::EulerFlux const numerical_flux = this->m_riemann_solver(cons, cons, 0, eos);
    novapp::EulerFlux const physical_flux = compute_flux(cons, 0, eos);
    EXPECT_DOUBLE_EQ(numerical_flux.density, physical_flux.density);
    EXPECT_DOUBLE_EQ(numerical_flux.momentum[0], physical_flux.momentum[0]);
    EXPECT_DOUBLE_EQ(numerical_flux.energy, physical_flux.energy);
}

TYPED_TEST(RiemannSolverFixture, Symmetry)
{
    novapp::thermodynamics::PerfectGas const eos(1.4, 1., 0);
    novapp::EulerPrim prim_left;
    prim_left.density = 2.;
    prim_left.velocity = {3.};
    prim_left.pressure = 10.;
    novapp::EulerPrim prim_right;
    prim_right.density = 3.;
    prim_right.velocity = {-6.};
    prim_right.pressure = 5.5;
    novapp::EulerCons cons_left = to_cons(prim_left, eos);
    novapp::EulerCons cons_right = to_cons(prim_right, eos);
    novapp::EulerFlux const flux1 = this->m_riemann_solver(cons_left, cons_right, 0, eos);
    std::swap(cons_left, cons_right);
    cons_left.momentum[0] = -cons_left.momentum[0];
    cons_right.momentum[0] = -cons_right.momentum[0];
    novapp::EulerFlux const flux2 = this->m_riemann_solver(cons_left, cons_right, 0, eos);
    EXPECT_DOUBLE_EQ(flux1.density, -flux2.density);
    EXPECT_DOUBLE_EQ(flux1.momentum[0], flux2.momentum[0]);
    EXPECT_DOUBLE_EQ(flux1.energy, -flux2.energy);
}
