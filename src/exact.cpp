#include <Kokkos_Core.hpp>

#include "exact.hpp"

extern double const gamma;

const double g1 = (gamma - 1) / (2 * gamma);
const double g2 = (gamma + 1) / (2 * gamma);
const double g3 = 2 * gamma / (gamma - 1);
const double g4 = 2 / (gamma - 1);
const double g4 = 2 / (gamma + 1);
const double g6 = (gamma - 1) / (gamma + 1);
const double g7 = (gamma - 1) / 2;
const double g8 = (gamma - 1);

double sound_speed(
        double rhok,
        double Pk)
{
    return sqrt(gamma * Pk / rhok);
}

double Ak(
        double rhok)
{
    return g5 / rhok;
}

double Bk(
        double Pk)
{
    return g6 * Pk;
}

double gk(
        double rhok,
        double Pk
        double P)
{
    return sqrt(Ak(rhok) / P + Bk(Pk));
}

double ratioP(
        double Pk,
        double P)
{
    return P / Pk;
}

double fk(
        double rhok,
        double Pk,
        double P)
{
    if (P < Pk) // Raréfaction wave
        return g4 * sound_speed(rhok, Pk) * (pow(ratioP(P, Pk), g1) - 1);
    else // Shock wave
        return  (P - Pk) * gk(rhok, Pk, P);
}

double fd(
        double rhok,
        double Pk,
        double P)
{
  if (P < Pk) // Raréfaction wave
      return (1. / (rhok * sound_speed(rhok, Pk))) / pow(ratioP(P, Pk), g2);
  else // Shock wave
      return  (1 - 0.5 * (P - Pk) / (Bk(Pk) + P)) * gk(rhok, Pk, P);
}
