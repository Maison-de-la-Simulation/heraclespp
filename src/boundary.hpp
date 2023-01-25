/**
 * @file boundary.hpp
 * Boundary condition
 */
#pragma once

#include <Kokkos_Core.hpp>

void GradientNull(
    Kokkos::View<double ***> const rho,
    Kokkos::View<double ***> const rhou,
    Kokkos::View<double ***> const E);

void Periodic(
    Kokkos::View<double ***> const rho,
    Kokkos::View<double ***> const rhou,
    Kokkos::View<double ***> const E);

void Reflexive(
    Kokkos::View<double ***> const rho,
    Kokkos::View<double ***> const rhou,
    Kokkos::View<double ***> const E);