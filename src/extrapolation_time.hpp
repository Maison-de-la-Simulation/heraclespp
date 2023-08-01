//!
//! @file face_reconstruction.hpp
//!

#pragma once

#include <type_traits>

#include <Kokkos_Core.hpp>

#include "eos.hpp"
#include "euler_equations.hpp"
#include "geom.hpp"
#include "grid.hpp"
#include "kokkos_shortcut.hpp"
#include "kronecker.hpp"
#include "ndim.hpp"
#include "range.hpp"

namespace novapp
{

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
        double dt_reconstruction,
        KV_double_5d rho_rec,
        KV_double_6d rhou_rec,
        KV_double_5d E_rec,
        KV_double_6d loc_u_rec,
        KV_double_5d loc_P_rec,
        KV_double_6d fx_rec) const
        = 0;
};

template <class Gravity>
class ExtrapolationTimeReconstruction : public IExtrapolationReconstruction
{
    static_assert(
            std::is_invocable_r_v<
                void,
                Gravity,
                int,
                int,
                int,
                int>,
            "Incompatible gravity.");

private:
    EOS m_eos;
    Grid m_grid;
    Gravity m_gravity;

public:
    ExtrapolationTimeReconstruction(
            EOS const& eos, 
            Grid const& grid,
            Gravity const& gravity)
        : m_eos(eos)
        , m_grid(grid)
        , m_gravity(gravity)
    {
    }

    void execute(
        Range const& range,
        double const dt_reconstruction,
        KV_double_5d const rho_rec,
        KV_double_6d const rhou_rec,
        KV_double_5d const E_rec,
        KV_double_6d const loc_u_rec,
        KV_double_5d const loc_P_rec,
        KV_double_6d const fx_rec) const final
    {
        assert(rho_rec.extent(0) == rhou_rec.extent(0));
        assert(rhou_rec.extent(0) == E_rec.extent(0));
        assert(rho_rec.extent(1) == rhou_rec.extent(1));
        assert(rhou_rec.extent(1) == E_rec.extent(1));
        assert(rho_rec.extent(2) == rhou_rec.extent(2));
        assert(rhou_rec.extent(2) == E_rec.extent(2));
        assert(rho_rec.extent(3) == rhou_rec.extent(3));
        assert(rhou_rec.extent(3) == E_rec.extent(3));
        assert(rho_rec.extent(4) == rhou_rec.extent(4));
        assert(rhou_rec.extent(4) == E_rec.extent(4));

        int nfx = fx_rec.extent_int(5);
        auto const xc = m_grid.x_center;
        auto const x = m_grid.x.d_view;
        auto const ds = m_grid.ds;
        auto const dv = m_grid.dv;

        KV_double_6d fx_rec_old("rho_rec_old", m_grid.Nx_local_wg[0], m_grid.Nx_local_wg[1], 
                                m_grid.Nx_local_wg[2], 2, ndim, nfx);
        Kokkos::deep_copy(fx_rec_old, fx_rec);

        auto const [begin, end] = cell_range(range);
        Kokkos::parallel_for(
            "Hancock_extrapolation",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
            KOKKOS_CLASS_LAMBDA(int i, int j, int k)
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

                    EulerPrim minus_one; // Left, front, bottom
                    minus_one.rho = rho_old[0][idim];
                    for (int idr = 0; idr < ndim; ++idr)
                    {
                        minus_one.u[idr] = loc_u_rec(i, j, k, 0, idim, idr);
                    }
                    minus_one.P = loc_P_rec(i, j, k, 0, idim);
                    EulerFlux const flux_minus_one = compute_flux(minus_one, idim, m_eos);

                    EulerPrim plus_one; // Right, back, top
                    plus_one.rho = rho_old[1][idim];
                    for (int idr = 0; idr < ndim; ++idr)
                    {
                        plus_one.u[idr] = loc_u_rec(i, j, k, 1, idim, idr);
                    }
                    plus_one.P = loc_P_rec(i, j, k, 1, idim);
                    EulerFlux const flux_plus_one = compute_flux(plus_one, idim, m_eos);

                    double const dtodv = dt_reconstruction / dv(i, j, k);

                    for (int ipos = 0; ipos < ndim; ++ipos)
                    {
                        rho_rec(i, j, k, 0, ipos) += dtodv * (flux_minus_one.rho * ds(i, j, k, idim) 
                                                    - flux_plus_one.rho * ds(i_p, j_p, k_p, idim));
                        rho_rec(i, j, k, 1, ipos) += dtodv * (flux_minus_one.rho * ds(i, j, k, idim) 
                                                    - flux_plus_one.rho * ds(i_p, j_p, k_p, idim));
                        for (int idr = 0; idr < ndim; ++idr)
                        {
                            rhou_rec(i, j, k, 0, ipos, idr) += dtodv * (flux_minus_one.rhou[idr] * ds(i, j, k, idim) 
                                                            - flux_plus_one.rhou[idr] * ds(i_p, j_p, k_p, idim));
                            rhou_rec(i, j, k, 1, ipos, idr) += dtodv * (flux_minus_one.rhou[idr] * ds(i, j, k, idim) 
                                                            - flux_plus_one.rhou[idr] * ds(i_p, j_p, k_p, idim));
                        }
                        E_rec(i, j, k, 0, ipos) += dtodv * (flux_minus_one.E * ds(i, j, k, idim) 
                                                - flux_plus_one.E * ds(i_p, j_p, k_p, idim));
                        E_rec(i, j, k, 1, ipos) += dtodv * (flux_minus_one.E * ds(i, j, k, idim) 
                                                - flux_plus_one.E * ds(i_p, j_p, k_p, idim));
                    }

                    //Spherical geometric terms
                    if (geom_choice == "SPHERICAL")
                    {
                        double rm = x(i);
                        double rp = x(i_p);

                        if (ndim == 1)
                        {
                            for (int ipos = 0; ipos < ndim; ++ipos)
                            {
                                rhou_rec(i, j, k, 0, ipos, idim) += dt_reconstruction * 3 * minus_one.P
                                                                    * (rp*rp - rm*rm) / (rp*rp*rp - rm*rm*rm);
                                rhou_rec(i, j, k, 1, ipos, idim) += dt_reconstruction * 3 * plus_one.P
                                                                    * (rp*rp - rm*rm) / (rp*rp*rp - rm*rm*rm);
                            }
                        }
                    }

                    // Gravity
                    for (int ipos = 0; ipos < ndim; ++ipos)
                    {
                        rhou_rec(i, j, k, 0, ipos, idim) += dt_reconstruction * m_gravity(i, j, k, idim) * rho_old[0][ipos];
                        rhou_rec(i, j, k, 1, ipos, idim) += dt_reconstruction * m_gravity(i, j, k, idim) * rho_old[1][ipos];
                        
                        E_rec(i, j, k, 0, ipos) += dt_reconstruction * m_gravity(i, j, k, idim) * rhou_old[0][ipos][idim];
                        E_rec(i, j, k, 1, ipos) += dt_reconstruction * m_gravity(i, j, k, idim) * rhou_old[1][ipos][idim];
                    }

                    // Passive scalar
                    for (int ifx = 0; ifx < nfx; ++ifx)
                    {
                        for (int ipos = 0; ipos < ndim; ++ipos)
                        {
                            double flux_fx_L = fx_rec_old(i, j, k, 0, idim, ifx) * flux_minus_one.rho;
                            double flux_fx_R = fx_rec_old(i, j, k, 1, idim, ifx) * flux_plus_one.rho;

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
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
            KOKKOS_CLASS_LAMBDA(int i, int j, int k)
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
