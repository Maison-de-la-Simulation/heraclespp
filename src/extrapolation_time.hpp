//!
//! @file face_reconstruction.hpp
//!

#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include <PerfectGas.hpp>

#include "ndim.hpp"
#include "range.hpp"
#include "euler_equations.hpp"
#include "grid.hpp"
#include "gravity.hpp"
#include "Kokkos_shortcut.hpp"
#include "kronecker.hpp"


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
        KV_double_5d rho_rec,
        KV_double_6d rhou_rec,
        KV_double_5d E_rec,
        KV_double_6d loc_u_rec,
        KV_double_5d loc_P_rec,
        double const dt,
        Grid const& grid) const
        = 0;
};

template <class Gravity>
class ExtrapolationTimeReconstruction : public IExtrapolationReconstruction
{
    /* static_assert(
            std::is_invocable_r_v<>,
            "Invalid gravity."); */

private:
    Gravity m_gravity;

    thermodynamics::PerfectGas m_eos;

public:
    ExtrapolationTimeReconstruction(Gravity const& gravity,
    thermodynamics::PerfectGas const& eos)
        : m_gravity(gravity)
        , m_eos(eos)
    {
    }

    void execute(
        Range const& range,
        KV_double_5d const rho_rec,
        KV_double_6d const rhou_rec,
        KV_double_5d const E_rec,
        KV_double_6d const loc_u_rec,
        KV_double_5d const loc_P_rec,
        double const dt,
        Grid const& grid) const final
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

        // Intermediate array
        KV_double_5d rho_rec_old("rho_rec_old", grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, ndim);
        KV_double_6d rhou_rec_old("rhou_rec_old", grid.Nx_local_wg[0], grid.Nx_local_wg[1], grid.Nx_local_wg[2], 2, ndim, ndim);
        Kokkos::deep_copy(rho_rec_old, rho_rec);
        Kokkos::deep_copy(rhou_rec_old, rhou_rec);

        auto const [begin, end] = cell_range(range);

        Kokkos::parallel_for(
        "HancockExtrapolation",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
        KOKKOS_CLASS_LAMBDA(int i, int j, int k)
        {
            for (int idim = 0; idim < ndim; ++idim)
            {
                auto const [i_p, j_p, k_p] = rindex(idim, i, j, k); // i + 1

                EulerPrim minus_one; // Left, front, bottom
                minus_one.rho = rho_rec(i, j, k, 0, idim);
                for (int idr = 0; idr < ndim; ++idr)
                {
                    minus_one.u[idr] = loc_u_rec(i, j, k, 0, idim, idr);
                }
                minus_one.P = loc_P_rec(i, j, k, 0, idim);
                EulerFlux flux_minus_one = compute_flux(minus_one, idim, m_eos);

                EulerPrim plus_one; // Right, back, top
                plus_one.rho = rho_rec(i, j, k, 1, idim);
                for (int idr = 0; idr < ndim; ++idr)
                {
                    plus_one.u[idr] = loc_u_rec(i, j, k, 1, idim, idr);
                }
                plus_one.P = loc_P_rec(i, j, k, 1, idim);
                EulerFlux flux_plus_one = compute_flux(plus_one, idim, m_eos);

                double dtodv = dt / grid.dv(i, j, k);

                for (int ipos = 0; ipos < ndim; ++ipos)
                {
                    rho_rec(i, j, k, 0, ipos) += dtodv * (flux_minus_one.rho * grid.ds(i, j, k, idim) - flux_plus_one.rho * grid.ds(i_p, j_p, k_p, idim));
                    rho_rec(i, j, k, 1, ipos) += dtodv * (flux_minus_one.rho * grid.ds(i, j, k, idim) - flux_plus_one.rho * grid.ds(i_p, j_p, k_p, idim));
                    for (int idr = 0; idr < ndim; ++idr)
                    {
                        rhou_rec(i, j, k, 0, ipos, idr) += dtodv * (flux_minus_one.rhou[idr] * grid.ds(i, j, k, idim) - flux_plus_one.rhou[idr] * grid.ds(i_p, j_p, k_p, idim));
                        rhou_rec(i, j, k, 1, ipos, idr) += dtodv * (flux_minus_one.rhou[idr] * grid.ds(i, j, k, idim) - flux_plus_one.rhou[idr] * grid.ds(i_p, j_p, k_p, idim));
                    }
                    E_rec(i, j, k, 0, ipos) += dtodv * (flux_minus_one.E * grid.ds(i, j, k, idim) - flux_plus_one.E * grid.ds(i_p, j_p, k_p, idim));
                    E_rec(i, j, k, 1, ipos) += dtodv * (flux_minus_one.E * grid.ds(i, j, k, idim) - flux_plus_one.E * grid.ds(i_p, j_p, k_p, idim));
                }
                
                // Gravity
                for (int ipos = 0; ipos < ndim; ++ipos)
                {
                    for (int idr = 0; idr < ndim; ++idr)
                    {
                        rhou_rec(i, j, k, 0, ipos, idr) += dt * m_gravity(i, j, k, idr, grid) * rho_rec_old(i, j, k, 0, ipos);
                        rhou_rec(i, j, k, 1, ipos, idr) += dt * m_gravity(i, j, k, idr, grid) * rho_rec_old(i, j, k, 1, ipos);
                        E_rec(i, j, k, 0, ipos) += dt * m_gravity(i, j, k, idr, grid) * rhou_rec_old(i, j, k, 0, ipos, idr);
                        E_rec(i, j, k, 1, ipos) += dt * m_gravity(i, j, k, idr, grid) * rhou_rec_old(i, j, k, 1, ipos, idr);
                    }
                }
            }
        });
    }
};

inline std::unique_ptr<IExtrapolationReconstruction> factory_time_reconstruction(
        std::string const& gravity,
        thermodynamics::PerfectGas const& eos,
        KV_double_1d &g)
{
    if (gravity == "Uniform")
    {
        return std::make_unique<ExtrapolationTimeReconstruction<UniformGravity>>(UniformGravity(g), eos);
    }
    throw std::runtime_error("Unknown time reconstruction algorithm: " + gravity + ".");
}

} // namespace novapp