// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file face_reconstruction.hpp
//!

#pragma once

#include <cassert>
#include <type_traits>

#include <Kokkos_Core.hpp>
#include <geom.hpp>
#include <grid.hpp>
#include <kokkos_shortcut.hpp>
#include <kronecker.hpp>
#include <ndim.hpp>
#include <range.hpp>

#include "concepts.hpp"
#include "euler_equations.hpp"
#include "source_terms.hpp"

namespace novapp
{

template <concepts::GravityField Gravity>
class IExtrapolationReconstruction
{
public:
    IExtrapolationReconstruction() = default;

    IExtrapolationReconstruction(IExtrapolationReconstruction const& rhs) = default;

    IExtrapolationReconstruction(IExtrapolationReconstruction&& rhs) noexcept = default;

    virtual ~IExtrapolationReconstruction() noexcept = default;

    IExtrapolationReconstruction& operator=(IExtrapolationReconstruction const& rhs) = default;

    IExtrapolationReconstruction& operator=(IExtrapolationReconstruction&& rhs) noexcept = default;

    virtual void execute(
        Range const& range,
        Grid const& grid,
        Gravity const& gravity,
        double dt_reconstruction,
        KV_cdouble_6d const& u_rec,
        KV_cdouble_5d const& P_rec,
        KV_double_5d const& rho_rec,
        KV_double_6d const& rhou_rec,
        KV_double_5d const& E_rec,
        KV_double_6d const& fx_rec) const
        = 0;
};

template <concepts::EulerEoS EoS, concepts::GravityField Gravity>
class ExtrapolationTimeReconstruction : public IExtrapolationReconstruction<Gravity>
{
private:
    EoS m_eos;

public:
    explicit ExtrapolationTimeReconstruction(EoS const& eos)
        : m_eos(eos)
    {
    }

    void execute(
        Range const& range,
        Grid const& grid,
        Gravity const& gravity,
        double const dt_reconstruction,
        KV_cdouble_6d const& u_rec,
        KV_cdouble_5d const& P_rec,
        KV_double_5d const& rho_rec,
        KV_double_6d const& rhou_rec,
        KV_double_5d const& E_rec,
        KV_double_6d const& fx_rec) const final
    {
        assert(equal_extents({0, 1, 2, 3, 4}, rho_rec, rhou_rec, E_rec, fx_rec, u_rec, P_rec));
        assert(equal_extents(5, rhou_rec, u_rec));

        auto const& eos = m_eos;
        int const nfx = fx_rec.extent_int(5);
        auto const x = grid.x;
        auto const y = grid.y;
        auto const ds = grid.ds;
        auto const dv = grid.dv;

        KV_double_6d const fx_rec_old("fx_rec_old", fx_rec.layout());
        Kokkos::deep_copy(fx_rec_old, fx_rec);

        Kokkos::parallel_for(
            "Hancock_extrapolation",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                Kokkos::Array<Kokkos::Array<double, ndim>, 2> rho_old; //rho_old[0/1][idim]
                Kokkos::Array<Kokkos::Array<Kokkos::Array<double, ndim>, ndim>, 2> rhou_old; //rhou_old[0/1][idim][idr]

                for (int idim = 0; idim < ndim; ++idim)
                {
                    rho_old[0][idim] = rho_rec(i, j, k, 0, idim);
                    rho_old[1][idim] = rho_rec(i, j, k, 1, idim);
                    for (int idr = 0; idr < ndim; ++idr)
                    {
                        rhou_old[0][idim][idr] = rhou_rec(i, j, k, 0, idim, idr);
                        rhou_old[1][idim][idr] = rhou_rec(i, j, k, 1, idim, idr);
                    }

                    for (int ifx = 0; ifx < nfx; ++ifx)
                    {
                        fx_rec(i, j, k, 0, idim, ifx) *= rho_rec(i, j, k, 0, idim);
                        fx_rec(i, j, k, 1, idim, ifx) *= rho_rec(i, j, k, 1, idim);
                    }
                }

                for (int idim = 0; idim < ndim; ++idim)
                {
                    auto const [i_p, j_p, k_p] = rindex(idim, i, j, k); // i + 1

                    EulerPrim primL; // Left, front, bottom
                    primL.rho = rho_old[0][idim];
                    for (int idr = 0; idr < ndim; ++idr)
                    {
                        primL.u[idr] = u_rec(i, j, k, 0, idim, idr);
                    }
                    primL.P = P_rec(i, j, k, 0, idim);
                    EulerFlux const fluxL = compute_flux(primL, idim, eos);

                    EulerPrim primR; // Right, back, top
                    primR.rho = rho_old[1][idim];
                    for (int idr = 0; idr < ndim; ++idr)
                    {
                        primR.u[idr] = u_rec(i, j, k, 1, idim, idr);
                    }
                    primR.P = P_rec(i, j, k, 1, idim);
                    EulerFlux const fluxR = compute_flux(primR, idim, eos);

                    double const dtodv = dt_reconstruction / dv(i, j, k);

                    for (int ipos = 0; ipos < ndim; ++ipos)
                    {
                        rho_rec(i, j, k, 0, ipos) += dtodv * (fluxL.rho * ds(i, j, k, idim)
                                                    - fluxR.rho * ds(i_p, j_p, k_p, idim));
                        rho_rec(i, j, k, 1, ipos) += dtodv * (fluxL.rho * ds(i, j, k, idim)
                                                    - fluxR.rho * ds(i_p, j_p, k_p, idim));
                        for (int idr = 0; idr < ndim; ++idr)
                        {
                            rhou_rec(i, j, k, 0, ipos, idr) += dtodv * (fluxL.rhou[idr] * ds(i, j, k, idim)
                                                            - fluxR.rhou[idr] * ds(i_p, j_p, k_p, idim));
                            rhou_rec(i, j, k, 1, ipos, idr) += dtodv * (fluxL.rhou[idr] * ds(i, j, k, idim)
                                                            - fluxR.rhou[idr] * ds(i_p, j_p, k_p, idim));
                        }
                        E_rec(i, j, k, 0, ipos) += dtodv * (fluxL.E * ds(i, j, k, idim)
                                                - fluxR.E * ds(i_p, j_p, k_p, idim));
                        E_rec(i, j, k, 1, ipos) += dtodv * (fluxL.E * ds(i, j, k, idim)
                                                - fluxR.E * ds(i_p, j_p, k_p, idim));
                    }

                    if (geom == Geometry::Geom_spherical)
                    {
                        for (int ipos = 0; ipos < ndim; ++ipos)
                        {
                            if (idim == 0)
                            {
                                // Pressure term (e_{r}): 2 * P_{rr} / r
                                rhou_rec(i, j, k, 0, ipos, idim) += source_grad_P(dtodv, primL.P, primR.P,
                                                                    ds(i, j, k, idim), ds(i_p, j_p, k_p, idim));
                                rhou_rec(i, j, k, 1, ipos, idim) += source_grad_P(dtodv, primL.P, primR.P,
                                                                    ds(i, j, k, idim), ds(i_p, j_p, k_p, idim));

                                if (ndim == 2 || ndim == 3)
                                {
                                    // Velocity term (e_{r}): rho * u_{th} * u_{th} / r
                                    rhou_rec(i, j, k, 0, ipos, idim) += source_grad_u_r(dtodv, primL.rho, primR.rho,
                                                                    primL.u[1], primR.u[1], ds(i, j, k, idim), ds(i_p, j_p, k_p, idim));
                                    rhou_rec(i, j, k, 1, ipos, idim) += source_grad_u_r(dtodv, primL.rho, primR.rho,
                                                                    primL.u[1], primR.u[1], ds(i, j, k, idim), ds(i_p, j_p, k_p, idim));

                                    // Velocity term (e_{th}): rho * u_{th} * u_{r} / r
                                    rhou_rec(i, j, k, 0, ipos, 1) -= source_grad_u_idir_r(dtodv, x(i), x(i+1), primL.rho, primR.rho,
                                                                    primL.u[0], primR.u[0], primL.u[1], primR.u[1],
                                                                    ds(i, j, k, idim), ds(i_p, j_p, k_p, idim));
                                    rhou_rec(i, j, k, 1, ipos, 1) -= source_grad_u_idir_r(dtodv, x(i), x(i+1), primL.rho, primR.rho,
                                                                    primL.u[0], primR.u[0], primL.u[1], primR.u[1],
                                                                    ds(i, j, k, idim), ds(i_p, j_p, k_p, idim));
                                }
                                if (ndim == 3)
                                {
                                    // Velocity term (e_{r}): rho * u_{phi} * u_{phi} / r
                                    rhou_rec(i, j, k, 0, ipos, idim) += source_grad_u_r(dtodv, primL.rho, primR.rho,
                                                                        primL.u[2], primR.u[2], ds(i, j, k, idim), ds(i_p, j_p, k_p, idim));
                                    rhou_rec(i, j, k, 1, ipos, idim) += source_grad_u_r(dtodv, primL.rho, primR.rho,
                                                                        primL.u[2], primR.u[2], ds(i, j, k, idim), ds(i_p, j_p, k_p, idim));

                                    // Velocity term (e_{phi}): rho * u_{phi} * u_{r} / r
                                    rhou_rec(i, j, k, 0, ipos, 2) -= source_grad_u_idir_r(dtodv, x(i), x(i+1), primL.rho, primR.rho,
                                                                    primL.u[0], primR.u[0], primL.u[2], primR.u[2],
                                                                    ds(i, j, k, idim), ds(i_p, j_p, k_p, idim));
                                    rhou_rec(i, j, k, 1, ipos, 2) -= source_grad_u_idir_r(dtodv, x(i), x(i+1), primL.rho, primR.rho,
                                                                    primL.u[0], primR.u[0], primL.u[2], primR.u[2],
                                                                    ds(i, j, k, idim), ds(i_p, j_p, k_p, idim));
                                }
                            }
                            if (idim == 1)
                            {
                                // Pressure term (e_{th}): cot(th) * P_{th th} / r
                                rhou_rec(i, j, k, 0, ipos, idim) += source_grad_P(dtodv, primL.P, primR.P,
                                                                    ds(i, j, k, idim), ds(i_p, j_p, k_p, idim));
                                rhou_rec(i, j, k, 1, ipos, idim) += source_grad_P(dtodv, primL.P, primR.P,
                                                                    ds(i, j, k, idim), ds(i_p, j_p, k_p, idim));

                                if (ndim == 3)
                                {
                                    // Velocity term (e_{th}): cot(th) * rho * u_{phi} * u_{phi} / r
                                    rhou_rec(i, j, k, 0, ipos, idim) += source_grad_u_th(dtodv, y(j), y(j+1),
                                                                        primL.rho, primR.rho, primL.u[2], primR.u[2],
                                                                        ds(i, j, k, idim), ds(i_p, j_p, k_p, idim));
                                    rhou_rec(i, j, k, 1, ipos, idim) += source_grad_u_th(dtodv, y(j), y(j+1),
                                                                        primL.rho, primR.rho, primL.u[2], primR.u[2],
                                                                        ds(i, j, k, idim), ds(i_p, j_p, k_p, idim));

                                    // Velocity term (e_{phi}): cot(th) * rho * u_{phi} * u_{th} / r
                                    rhou_rec(i, j, k, 0, ipos, 2) -= source_grad_u_phi(dtodv, y(j), y(j+1),
                                                                    primL.rho, primR.rho, primL.u[1], primR.u[1],
                                                                    primL.u[2], primR.u[2],
                                                                    ds(i, j, k, idim), ds(i_p, j_p, k_p, idim));
                                    rhou_rec(i, j, k, 1, ipos, 2) -= source_grad_u_phi(dtodv, y(j), y(j+1),
                                                                    primL.rho, primR.rho, primL.u[1], primR.u[1],
                                                                    primL.u[2], primR.u[2],
                                                                    ds(i, j, k, idim), ds(i_p, j_p, k_p, idim));
                                }
                            }
                        }
                    }

                    // Gravity
                    for (int ipos = 0; ipos < ndim; ++ipos)
                    {
                        rhou_rec(i, j, k, 0, ipos, idim) += dt_reconstruction * gravity(i, j, k, idim) * rho_old[0][ipos];
                        rhou_rec(i, j, k, 1, ipos, idim) += dt_reconstruction * gravity(i, j, k, idim) * rho_old[1][ipos];

                        E_rec(i, j, k, 0, ipos) += dt_reconstruction * gravity(i, j, k, idim) * rhou_old[0][ipos][idim];
                        E_rec(i, j, k, 1, ipos) += dt_reconstruction * gravity(i, j, k, idim) * rhou_old[1][ipos][idim];
                    }

                    // Passive scalar
                    for (int ifx = 0; ifx < nfx; ++ifx)
                    {
                        for (int ipos = 0; ipos < ndim; ++ipos)
                        {
                            double const flux_fx_L = fx_rec_old(i, j, k, 0, idim, ifx) * fluxL.rho;
                            double const flux_fx_R = fx_rec_old(i, j, k, 1, idim, ifx) * fluxR.rho;

                            fx_rec(i, j, k, 0, ipos, ifx) += dtodv * (flux_fx_L * ds(i, j, k, idim)
                                                                - flux_fx_R * ds(i_p, j_p, k_p, idim));
                            fx_rec(i, j, k, 1, ipos, ifx) += dtodv * (flux_fx_L * ds(i, j, k, idim)
                                                                - flux_fx_R * ds(i_p, j_p, k_p, idim));
                        }
                    }
                }
            });

        Kokkos::parallel_for(
            "passive_scalar_extrapolation",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                for (int idim = 0; idim < ndim; ++idim)
                {
                    for (int ifx = 0; ifx < nfx; ++ifx)
                    {
                        fx_rec(i, j, k, 0, idim, ifx) /= rho_rec(i, j, k, 0, idim);
                        fx_rec(i, j, k, 1, idim, ifx) /= rho_rec(i, j, k, 1, idim);
                    }
                }
            });
    }
};

} // namespace novapp
