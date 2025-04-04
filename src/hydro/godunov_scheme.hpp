// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file godunov_scheme.hpp
//!

#pragma once

#include <cassert>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <Kokkos_Core.hpp>
#include <geom.hpp>
#include <grid.hpp>
#include <kokkos_shortcut.hpp>
#include <kronecker.hpp>
#include <ndim.hpp>
#include <range.hpp>

#include "euler_equations.hpp"
#include "riemann_solver.hpp"
#include "source_terms.hpp"

namespace novapp
{

template <class Gravity>
class IGodunovScheme
{
public:
    IGodunovScheme() = default;

    IGodunovScheme(IGodunovScheme const& rhs) = default;

    IGodunovScheme(IGodunovScheme&& rhs) noexcept = default;

    virtual ~IGodunovScheme() noexcept = default;

    IGodunovScheme& operator=(IGodunovScheme const& rhs) = default;

    IGodunovScheme& operator=(IGodunovScheme&& rhs) noexcept = default;

    virtual void execute(
        Range const &range,
        Grid const& grid,
        Gravity const& gravity,
        double dt,
        KV_cdouble_3d const& rho,
        KV_cdouble_4d const& rhou,
        KV_cdouble_3d const& E,
        KV_cdouble_4d const& fx,
        KV_cdouble_5d const& rho_rec,
        KV_cdouble_6d const& rhou_rec,
        KV_cdouble_5d const& E_rec,
        KV_cdouble_6d const& fx_rec,
        KV_double_3d const& rho_new,
        KV_double_4d const& rhou_new,
        KV_double_3d const& E_new,
        KV_double_4d const& fx_new) const = 0;
};

template <class RiemannSolver, class Gravity, class EoS>
class RiemannBasedGodunovScheme : public IGodunovScheme<Gravity>
{
    static_assert(
            std::is_invocable_r_v<
                EulerFlux,
                RiemannSolver,
                EulerCons,
                EulerCons,
                int,
                EoS>,
            "Incompatible Riemann solver.");

private:
    RiemannSolver m_riemann_solver;
    EoS m_eos;

public:
    RiemannBasedGodunovScheme(
        RiemannSolver const &riemann_solver,
        EoS const& eos)
        : m_riemann_solver(riemann_solver)
        , m_eos(eos)
    {
    }

    void execute(
        Range const& range,
        Grid const& grid,
        Gravity const& gravity,
        double const dt,
        KV_cdouble_3d const& rho,
        KV_cdouble_4d const& rhou,
        KV_cdouble_3d const& E,
        KV_cdouble_4d const& fx,
        KV_cdouble_5d const& rho_rec,
        KV_cdouble_6d const& rhou_rec,
        KV_cdouble_5d const& E_rec,
        KV_cdouble_6d const& fx_rec,
        KV_double_3d const& rho_new,
        KV_double_4d const& rhou_new,
        KV_double_3d const& E_new,
        KV_double_4d const& fx_new) const final
    {
        assert(equal_extents({0, 1, 2}, rho, rhou, E, fx, rho_new, rhou_new, E_new, fx_new));
        assert(equal_extents({0, 1, 2, 3, 4}, rho_rec, rhou_rec, E_rec, fx_rec));
        assert(equal_extents({0, 1, 2}, rho, rho_rec));
        assert(equal_extents(3, rhou, rhou_new));
        assert(rhou_rec.extent(5) == ndim);
        assert(rhou.extent(3) == ndim);

        int const nfx = fx.extent_int(3);
        auto const x = grid.x;
        auto const y = grid.y;
        auto const ds = grid.ds;
        auto const dv = grid.dv;
        auto const& eos = m_eos;
        auto const& riemann_solver = m_riemann_solver;

        Kokkos::parallel_for(
            "Riemann_based_Godunov_scheme",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                rho_new(i, j, k) = rho(i, j, k);
                E_new(i, j, k) = E(i, j, k);

                for (int idim = 0; idim < ndim; ++idim)
                {
                    rhou_new(i, j, k, idim) = rhou(i, j, k, idim);
                }

                for (int ifx = 0; ifx < nfx; ++ifx)
                {
                    fx_new(i, j, k, ifx) = fx(i, j, k, ifx) * rho(i, j,k);
                }

                for (int idim = 0; idim < ndim; ++idim)
                {
                    auto const [i_m, j_m, k_m] = lindex(idim, i, j, k); // i - 1
                    auto const [i_p, j_p, k_p] = rindex(idim, i, j, k); // i + 1

                    EulerCons minus_oneR; // Right, back, top (i,j,k) - 1
                    minus_oneR.rho = rho_rec(i_m, j_m, k_m, 1, idim);
                    for (int idr = 0; idr < ndim; ++idr)
                    {
                        minus_oneR.rhou[idr] = rhou_rec(i_m, j_m, k_m, 1, idim, idr);
                    }
                    minus_oneR.E = E_rec(i_m, j_m, k_m, 1, idim);
                    EulerCons var_L; // Left, front, down (i,j,k)
                    var_L.rho = rho_rec(i, j, k, 0, idim);
                    for (int idr = 0; idr < ndim; ++idr)
                    {
                        var_L.rhou[idr] = rhou_rec(i, j, k, 0, idim, idr);
                    }
                    var_L.E = E_rec(i, j, k, 0, idim);
                    EulerFlux const FluxL = riemann_solver(minus_oneR, var_L, idim, eos);

                    EulerCons var_R; // Right, back, top (i,j,k)
                    var_R.rho = rho_rec(i, j, k, 1, idim);
                    for (int idr = 0; idr < ndim; ++idr)
                    {
                        var_R.rhou[idr] = rhou_rec(i, j, k, 1, idim, idr);
                    }
                    var_R.E = E_rec(i, j, k, 1, idim);
                    EulerCons plus_oneL; // Left, front, down (i,j,k) + 1
                    plus_oneL.rho = rho_rec(i_p, j_p, k_p, 0, idim);
                    for (int idr = 0; idr < ndim; ++idr)
                    {
                        plus_oneL.rhou[idr] = rhou_rec(i_p, j_p, k_p, 0, idim, idr);
                    }
                    plus_oneL.E = E_rec(i_p, j_p, k_p, 0, idim);
                    EulerFlux const FluxR = riemann_solver(var_R, plus_oneL, idim, eos);

                    double const dtodv = dt / dv(i, j, k);

                    rho_new(i, j, k) += dtodv * (FluxL.rho * ds(i, j, k, idim)
                                        - FluxR.rho * ds(i_p, j_p, k_p, idim));
                    for (int idr = 0; idr < ndim; ++idr)
                    {
                        rhou_new(i, j, k, idr) += dtodv * (FluxL.rhou[idr] * ds(i, j, k, idim)
                                                    - FluxR.rhou[idr] * ds(i_p, j_p, k_p, idim));
                    }
                    E_new(i, j, k) += dtodv * (FluxL.E * ds(i, j, k, idim)
                                            - FluxR.E * ds(i_p, j_p, k_p, idim));

                    if (geom == Geometry::Geom_spherical)
                    {
                        EulerPrim const primL = to_prim(var_L, eos);
                        EulerPrim const primR = to_prim(var_R, eos);

                        if (idim == 0)
                        {
                            // Pressure term (e_{r}): 2 * P_{rr} / r
                            rhou_new(i, j, k, idim) += source_grad_P(dtodv, primL.P, primR.P,
                                                    ds(i, j, k, idim), ds(i_p, j_p, k_p, idim));

                            if (ndim == 2 || ndim == 3)
                            {
                                // Velocity term (e_{r}): rho * u_{th} * u_{th} / r
                                rhou_new(i, j, k, idim) += source_grad_u_r(dtodv, primL.rho, primR.rho,
                                            primL.u[1], primR.u[1], ds(i, j, k, idim), ds(i_p, j_p, k_p, idim));

                                // Velocity term (e_{th}): rho * u_{th} * u_{r} / r
                                rhou_new(i, j, k, 1) -= source_grad_u_idir_r(dtodv, x(i), x(i+1), primL.rho, primR.rho,
                                                        primL.u[0], primR.u[0], primL.u[1], primR.u[1],
                                                        ds(i, j, k, idim), ds(i_p, j_p, k_p, idim));
                            }
                            if (ndim == 3)
                            {
                                // Velocity term (e_{r}): rho * u_{phi} * u_{phi} / r
                                rhou_new(i, j, k, idim) += source_grad_u_r(dtodv, primL.rho, primR.rho,
                                                        primL.u[2], primR.u[2], ds(i, j, k, idim), ds(i_p, j_p, k_p, idim));

                                // Velocity term (e_{phi}): rho * u_{phi} * u_{r} / r
                                rhou_new(i, j, k, 2) -= source_grad_u_idir_r(dtodv, x(i), x(i+1), primL.rho, primR.rho,
                                                    primL.u[0], primR.u[0], primL.u[2], primR.u[2],
                                                    ds(i, j, k, idim), ds(i_p, j_p, k_p, idim));
                            }
                        }
                        if (idim == 1)
                        {
                            // Pressure term (e_{th}): cot(th) * P_{th th} / r
                            rhou_new(i, j, k, idim) += source_grad_P(dtodv, primL.P, primR.P,
                                                    ds(i, j, k, idim), ds(i_p, j_p, k_p, idim));

                            if (ndim == 3)
                            {
                                // Velocity term (e_{th}): cot(th) * rho * u_{phi} * u_{phi} / r
                                rhou_new(i, j, k, idim) += source_grad_u_th(dtodv, y(j), y(j+1),
                                                        primL.rho, primR.rho, primL.u[2], primR.u[2],
                                                        ds(i, j, k, idim), ds(i_p, j_p, k_p, idim));

                                // Velocity term (e_{phi}): cot(th) * rho * u_{phi} * u_{th} / r
                                rhou_new(i, j, k, 2) -= source_grad_u_phi(dtodv, y(j), y(j+1),
                                                    primL.rho, primR.rho, primL.u[1], primR.u[1],
                                                    primL.u[2], primR.u[2],
                                                    ds(i, j, k, idim), ds(i_p, j_p, k_p, idim));
                            }
                        }
                    }

                    // Gravity
                    rhou_new(i, j, k, idim) += dt * gravity(i, j, k, idim) * rho(i, j, k);
                    E_new(i, j, k) += dt * gravity(i, j, k, idim) * rhou(i, j, k, idim);

                    // Passive scalar
                    for (int ifx = 0; ifx < nfx; ++ifx)
                    {
                        int iL_uw = i_m; // upwind
                        int iR_uw = i;
                        int jL_uw = j_m;
                        int jR_uw = j;
                        int kL_uw = k_m;
                        int kR_uw = k;
                        int face_L = 1;
                        int face_R = 1;

                        if (FluxL.rho < 0)
                        {
                            iL_uw = i;
                            jL_uw = j;
                            kL_uw = k;
                            face_L = 0;
                        }
                        if (FluxR.rho < 0)
                        {
                            iR_uw = i_p;
                            jR_uw = j_p;
                            kR_uw = k_p;
                            face_R = 0;
                        }

                        double const flux_fx_L = fx_rec(iL_uw, jL_uw, kL_uw, face_L, idim, ifx) * FluxL.rho;
                        double const flux_fx_R = fx_rec(iR_uw, jR_uw, kR_uw, face_R, idim, ifx) * FluxR.rho;

                        fx_new(i, j, k, ifx) += dtodv * (flux_fx_L * ds(i, j, k, idim)
                                                - flux_fx_R * ds(i_p, j_p, k_p, idim));
                    }
                }
            });

        Kokkos::parallel_for(
            "passive_scalar_Godunov",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                for (int ifx = 0; ifx < nfx; ++ifx)
                {
                    fx_new(i, j, k, ifx) = Kokkos::clamp(fx_new(i, j, k, ifx) / rho_new(i, j, k), 0., 1.);
                }
            });
    }
};

template <class EoS, class Gravity>
inline std::unique_ptr<IGodunovScheme<Gravity>> factory_godunov_scheme(
    std::string const& riemann_solver,
    EoS const& eos)
{
    if (riemann_solver == "HLL")
    {
        return std::make_unique<RiemannBasedGodunovScheme<HLL, Gravity, EoS>>
                (HLL(), eos);
    }

    if (riemann_solver == "HLLC")
    {
        return std::make_unique<RiemannBasedGodunovScheme<HLLC, Gravity, EoS>>
                (HLLC(), eos);
    }

    if (riemann_solver == "Low Mach")
    {
        return std::make_unique<RiemannBasedGodunovScheme<Splitting, Gravity, EoS>>
                (Splitting(), eos);
    }

    throw std::runtime_error("Invalid riemann solver: " + riemann_solver + ".");
}

} // namespace novapp
