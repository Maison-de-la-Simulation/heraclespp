// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <Kokkos_Core.hpp>
#include <kokkos_shortcut.hpp>
#include <ndim.hpp>
#include <range.hpp>

#include "concepts.hpp"
#include "euler_equations.hpp"
#include "grid.hpp"
#include "kronecker.hpp"

namespace novapp
{

template <concepts::EulerEoS EoS>
void pressure_fix(
    Range const& range,
    EoS const& eos,
    Grid const& grid,
    double const dt,
    double const eps,
    KV_cdouble_3d const& rho,
    KV_cdouble_4d const& rhou,
    KV_cdouble_3d const& E,
    KV_cdouble_5d const& rho_rec,
    KV_cdouble_6d const& rhou_rec,
    KV_cdouble_5d const& E_rec,
    KV_cdouble_3d const& rho_new,
    KV_cdouble_4d const& rhou_new,
    KV_double_3d const& E_new)
{
    auto const ds = grid.ds;
    auto const dv = grid.dv;

    Kokkos::parallel_for(
        "pressure_fix",
        cell_mdrange(range),
        KOKKOS_LAMBDA(int i, int j, int k)
        {
            EulerCons cons; // (i,j,k)
            EulerCons cons_new; // (i,j,k) t^{n+1}

            cons.rho = rho(i, j, k);
            for (int idr = 0; idr < ndim; ++idr)
            {
                cons.rhou[idr] = rhou(i, j, k, idr);
            }
            cons.E = E(i, j, k);

            cons_new.rho = rho_new(i, j, k);
            for (int idr = 0; idr < ndim; ++idr)
            {
                cons_new.rhou[idr] = rhou_new(i, j, k, idr);
            }
            cons_new.E = E_new(i, j, k);

            EulerPrim const prim = to_prim(cons, eos);

            //--------------------------------------------------------//

            double const ekin = compute_ek(cons); // Kinetic energy (i, j, k)
            double const evol = E(i, j, k) - ekin; // Internal energy (i, j, k) at n
            double const evol_new = E_new(i, j, k) - compute_ek(cons_new); // Internal energy (i, j, k) at n+1
            double const de_god = evol_new - evol;

            if (evol < eps * ekin)
            {
                E_new(i, j, k) = E(i, j, k);
                double divU = 0;
                double divUE = 0;
                double einL = 0;
                double einR = 0;
                double alpha = (evol * evol) / (ekin * ekin * eps * eps);

                if (evol < (eps / 10) * ekin)
                {
                    alpha = 0;
                }

                for (int idim = 0; idim < ndim; ++idim)
                {
                    auto const [i_m, j_m, k_m] = lindex(idim, i, j, k); // i - 1
                    auto const [i_p, j_p, k_p] = rindex(idim, i, j, k); // i + 1

                    for (int ipos = 0; ipos < ndim; ++ipos)
                    {
                        EulerCons var_L; // Left, front, down (i,j,k)
                        EulerCons var_R; // Right, back, top (i,j,k)

                        var_L.rho = rho_rec(i, j, k, 0, idim);
                        for (int idr = 0; idr < ndim; ++idr)
                        {
                            var_L.rhou[idr] = rhou_rec(i, j, k, 0, idim, idr);
                        }
                        var_L.E = E_rec(i, j, k, 0, idim);

                        var_R.rho = rho_rec(i, j, k, 1, idim);
                        for (int idr = 0; idr < ndim; ++idr)
                        {
                            var_R.rhou[idr] = rhou_rec(i, j, k, 1, idim, idr);
                        }
                        var_R.E = E_rec(i, j, k, 1, idim);

                        EulerPrim const primL = to_prim(var_L, eos);
                        EulerPrim const primR = to_prim(var_R, eos);

                        EulerCons minus_oneR; // Right, back, top (i,j,k) - 1
                        EulerCons plus_oneL; // Left, front, down (i,j,k) + 1

                        minus_oneR.rho = rho_rec(i_m, j_m, k_m, 1, idim);
                        for (int idr = 0; idr < ndim; ++idr)
                        {
                            minus_oneR.rhou[idr] = rhou_rec(i_m, j_m, k_m, 1, idim, idr);
                        }
                        minus_oneR.E = E_rec(i_m, j_m, k_m, 1, idim);

                        plus_oneL.rho = rho_rec(i_p, j_p, k_p, 0, idim);
                        for (int idr = 0; idr < ndim; ++idr)
                        {
                            plus_oneL.rhou[idr] = rhou_rec(i_p, j_p, k_p, 0, idim, idr);
                        }
                        plus_oneL.E = E_rec(i_p, j_p, k_p, 0, idim);

                        EulerPrim const prim_mR = to_prim(minus_oneR, eos);
                        EulerPrim const prim_pL = to_prim(plus_oneL, eos);

                        EulerCons cons_p; // (i,j,k) + 1
                        EulerCons cons_m; // (i,j,k) - 1

                        cons_p.rho = rho(i_p, j_p, k_p);
                        for (int idr = 0; idr < ndim; ++idr)
                        {
                            cons_p.rhou[idr] = rhou(i_p, j_p, k_p, idr);
                        }
                        cons_p.E = E(i_p, j_p, k_p);

                        cons_m.rho = rho(i_m, j_m, k_m);
                        for (int idr = 0; idr < ndim; ++idr)
                        {
                            cons_m.rhou[idr] = rhou(i_m, j_m, k_m, idr);
                        }
                        cons_m.E = E(i_m, j_m, k_m);

                        //--------------------------------------------//

                        double const us = (prim_mR.u[ipos] + primL.u[ipos]) / 2; // Not the values from the Riemann solver
                        double const us_p = (primR.u[ipos] + prim_pL.u[ipos]) / 2;

                        divU += (us_p * ds(i_p, j_p, k_p, ipos)
                            - us * ds(i, j, k, ipos)) / dv(i, j, k);

                        if (us_p > 0)
                        {
                            einR = E(i, j, k) - compute_ek(cons);
                        }
                        else
                        {
                            einR = E(i_p, j_p, k_p) - compute_ek(cons_p);
                        }

                        if (us < 0)
                        {
                            einL = E(i, j, k) - compute_ek(cons);
                        }
                        else
                        {
                            einL = E(i_m, j_m, k_m) - compute_ek(cons_m);
                        }

                        divUE += (us_p * einR * ds(i_p, j_p, k_p, ipos)
                                - us * einL * ds(i, j, k, ipos)) / dv(i, j, k);

                        double const de_pf = dt * (-prim.P * divU - divUE);

                        double sum = 0;
                        for (int idr = 0; idr < ndim; ++idr)
                        {
                            sum += cons.rhou[idr] * cons.rhou[idr] / cons.rho;
                        }

                        double const de_tot = alpha * de_god + (1 - alpha) * de_pf;

                        E_new(i, j, k) = evol + de_tot + 1. / 2 * sum;
                    }
                }
            }
        });
}

} // namespace novapp
