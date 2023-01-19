#include <Kokkos_Core.hpp>

#include "fonc.hpp"

void ConvPrimCons(
    Kokkos::View<double*> const rho,
    Kokkos::View<double*> const rhou,
    Kokkos::View<double*> const E,
    Kokkos::View<double*> const u,
    Kokkos::View<double*> const P,
    double const gamma)
{
    Kokkos::parallel_for(
        "ConvPrimCons",
        rho.extent(0),
        KOKKOS_LAMBDA(int i) {
            rhou(i) = rho(i) * u(i) ;
            E(i) = (1. / 2) * rho(i) * u(i) * u(i) + P(i) / (gamma - 1);
        });
}

void ConvConsPrim(
    Kokkos::View<double*> const rho,
    Kokkos::View<double*> const rhou,
    Kokkos::View<double*> const E,
    Kokkos::View<double*> const u,
    Kokkos::View<double*> const P,
    double const gamma)
{
    Kokkos::parallel_for(
        "ConvConsPrim",
        rho.extent(0),
        KOKKOS_LAMBDA(int i) {
            u(i) = rhou(i) / rho(i);
            P(i) = (gamma - 1) * (E(i) - (1. / 2) * rho(i) * u(i) * u(i));
        });
}
