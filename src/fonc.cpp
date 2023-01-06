#include <Kokkos_Core.hpp>

#include "fonc.hpp"

void ConvPrimCons(
        Kokkos::View<double*> const rho,
        Kokkos::View<double*> const rhou,
        Kokkos::View<double*> const E,
        Kokkos::View<double*> const u,
        Kokkos::View<double*> const P,
        int const nx,
        double const gamma)
{
    Kokkos::parallel_for(
            "ConvPrimCons",
            nx,
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
        int const nx,
        double const gamma)
{
    Kokkos::parallel_for(
            "ConvConsPrim",
            nx,
            KOKKOS_LAMBDA(int i) {
              u(i) = rhou(i) / rho(i);
              P(i) = (gamma - 1) * (E(i) - (1. / 2) *rho(i) * u(i) * u(i);
            });
}

double VanLeer(
        double DiffL,
        double DiffR)
{
    double slope;
    if(DiffL * DiffR <= 0)
    {
      slope = 0;
    }
    else
    {
      double R = DiffL / DiffR;
      double slope = (1. / 2) * (DiffR + DiffL) * (4 * R) / ((R + 1) * (R + 1));
    }
    return slope;
}

double Minmod(
        double DiffL,
        double DiffR)
{
    double slope;
    if(DiffL * DiffR <= 0)
    {
      slope = 0;
    }
    else
    {
      double slope = 1 / ((1 / DiffL) + (1 / DiffR));
    }
    return slope;
}

double VanAlbada(
        double DiffL,
        double DiffR)
{
    double slope;
    if(DiffL * DiffR <= 0)
    {
      slope = 0;
    }
    else
    {
      double R = DiffL / DiffR;
      double slope = (1. / 2) * (DiffR + DiffL) * (2 * R) / (R * R + 1);
    }
    return slope;
}

double FluxRho(
        double const rho,
        double const u)
{
    return rho * u;
}

double FluxRhou(
        double const rho,
        double const u,
        double const P)
{
    return rho * u * u + P;
}

double FluxE(
        double const rho,
        double const u,
        double const P,
        double const gamma)
{
    return u * (1. / 2) * rho * u * u + P / (gamma - 1) + P;
}
