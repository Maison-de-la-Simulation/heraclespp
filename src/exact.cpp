#include <Kokkos_Core.hpp>

#include "exact.hpp"
#include "global_var.hpp"
using namespace GV;


double sound_speed(
        double rhok,
        double Pk)
{
    return sqrt(GV::gamma * Pk / rhok);
}

double Ak(
        double rhok)
{
    return GV::g5 / rhok;
}

double Bk(
        double Pk)
{
    return GV::g6 * Pk;
}

double gk(
        double rhok,
        double Pk,
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
