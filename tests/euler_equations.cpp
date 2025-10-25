// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Kokkos_Array.hpp>
#include <PerfectGas.hpp>
#include <euler_equations.hpp>

TEST(EulerEquations, KineticEnergyFromPrim)
{
    double const rho0 = 2.;
    double const u0 = 10.;
    double const P0 = 0.;

    novapp::EulerPrim const prim {.rho = rho0, .u = {u0}, .P = P0};
    EXPECT_DOUBLE_EQ(compute_ek(prim), 100.);
}

TEST(EulerEquations, KineticEnergyFromCons)
{
    double const rho0 = 2.;
    double const rhou0 = 20.;
    double const E0 = 100.;

    novapp::EulerCons const cons {.rho = rho0, .rhou = {rhou0}, .E = E0};
    EXPECT_DOUBLE_EQ(compute_ek(cons), 100.);
}

TEST(EulerEquations, FluxFromPrim)
{
    double const gamma = 1.4;
    double const mmw = 1.;
    double const rho0 = 2.;
    double const u0 = 10.;
    double const P0 = 0.;
    int const locdim = 0;

    novapp::thermodynamics::PerfectGas const eos(gamma, mmw);
    novapp::EulerPrim const prim {.rho = rho0, .u = {u0}, .P = P0};
    novapp::EulerFlux const flux = novapp::compute_flux(prim, locdim, eos);
    EXPECT_DOUBLE_EQ(flux.rho, 20.);
    EXPECT_DOUBLE_EQ(flux.E, 1000.);
    EXPECT_DOUBLE_EQ(flux.rhou[0], 200.);
}

TEST(EulerEquations, FluxFromCons)
{
    double const gamma = 1.4;
    double const mmw = 1.;
    double const rho0 = 2.;
    double const rhou0 = 20.;
    double const E0 = 100.;
    int const locdim = 0;

    novapp::thermodynamics::PerfectGas const eos(gamma, mmw);
    novapp::EulerCons const cons {.rho = rho0, .rhou = {rhou0}, .E = E0};
    novapp::EulerFlux const flux = novapp::compute_flux(cons, locdim, eos);
    EXPECT_DOUBLE_EQ(flux.rho, 20.);
    EXPECT_DOUBLE_EQ(flux.E, 1000.);
    EXPECT_DOUBLE_EQ(flux.rhou[0], 200.);
}

TEST(EulerEquations, PrimToConsToPrim)
{
    double const gamma = 1.4;
    double const mmw = 1.;
    double const rho0 = 2.;
    double const u0 = 10.;
    double const P0 = 1.;

    novapp::thermodynamics::PerfectGas const eos(gamma, mmw);
    novapp::EulerPrim const prim {.rho = rho0, .u = {u0}, .P = P0};
    novapp::EulerPrim const prim_to_cons_to_prim = to_prim(to_cons(prim, eos), eos);
    EXPECT_DOUBLE_EQ(prim.rho, prim_to_cons_to_prim.rho);
    EXPECT_DOUBLE_EQ(prim.P, prim_to_cons_to_prim.P);
    EXPECT_DOUBLE_EQ(prim.u[0], prim_to_cons_to_prim.u[0]);
}
