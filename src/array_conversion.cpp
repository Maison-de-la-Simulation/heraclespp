#include <Kokkos_Core.hpp>

#include "array_conversion.hpp"

void ConvPrimConsArray(
    Kokkos::View<double***> const rho,
    Kokkos::View<double***> const rhou,
    Kokkos::View<double***> const E,
    Kokkos::View<double***> const u,
    Kokkos::View<double***> const P,
    double const gamma)
{
    Kokkos::parallel_for(
        "ConvPrimConsArray",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
        {0, 0, 0},
        {rho.extent(0), rho.extent(1), rho.extent(2)}),
        KOKKOS_LAMBDA(int i, int j, int k)
    {
            rhou(i, j, k) =  rho(i, j, k) * u(i, j, k);
            E(i, j, k) = (1. / 2) * rho(i, j, k) * u(i, j, k) * u(i, j, k) + P(i, j, k) / (gamma - 1);
    });
}

void ConvConsPrimArray(
    Kokkos::View<double***> const rho,
    Kokkos::View<double***> const rhou,
    Kokkos::View<double***> const E,
    Kokkos::View<double***> const u,
    Kokkos::View<double***> const P,
    double const gamma)
{
    Kokkos::parallel_for(
        "ConvConsPrimArray",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
        {0, 0, 0},
        {rho.extent(0), rho.extent(1), rho.extent(2)}),
        KOKKOS_LAMBDA(int i, int j, int k)
        {
            u(i, j, k) = rhou(i, j, k) / rho(i, j, k);
            P(i, j, k) = (gamma - 1) * (E(i, j, k) - (1. / 2) * rho(i, j, k) * u(i, j, k) * u(i, j, k));
        });
}