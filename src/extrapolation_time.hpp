//!
//! @file face_reconstruction.hpp
//!

#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include "ndim.hpp"
#include "range.hpp"
#include "eos.hpp"
#include "euler_equations.hpp"
#include "grid.hpp"
#include "gravity.hpp"
#include "kokkos_shortcut.hpp"
#include "kronecker.hpp"
#include "nova_params.hpp"

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
        double const dt_reconstruction,
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
            std::is_invocable_r_v<void, Gravity, int, int, int, int, Grid>,
            "Incompatible gravity.");

private:
    Gravity m_gravity;

    EOS m_eos;

    Grid m_grid;

public:
    ExtrapolationTimeReconstruction(
            Gravity const& gravity,
            EOS const& eos, 
            Grid const& grid)
        : m_gravity(gravity)
        , m_eos(eos)
        , m_grid(grid)
    {
    }

    void execute(
        Range const& range,
        double const dt,
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

        auto const [begin, end] = cell_range(range);
        my_parallel_for(
            begin, end,
            KOKKOS_CLASS_LAMBDA(int i, int j, int k) KOKKOS_IMPL_HOST_FORCEINLINE
            {
                Kokkos::Array<double, 2*ndim> dsodv_alloc;
                Kokkos::View<double[2][ndim], Kokkos::LayoutRight> dsodv(dsodv_alloc.data());

                double const invdv = 1 / m_grid.dv(i, j, k);

                NOVA_FORCEUNROLL
                for (int idim = 0; idim < ndim; ++idim) {
                    auto const [i_p, j_p, k_p] = rindex(idim, i, j, k); // i + 1
                    dsodv(0, idim) = m_grid.ds(i, j, k, idim) * invdv;
                    dsodv(1, idim) = m_grid.ds(i_p, j_p, k_p, idim) * invdv;
                }

                EulerFlux flux{};
                Kokkos::Array<double, nfx> flux_fx{};

                NOVA_FORCEUNROLL
                for (int idim = 0; idim < ndim; ++idim) {
                    EulerPrim minus_one; // Left, front, bottom
                    minus_one.rho = rho_rec(i, j, k, 0, idim);
                    NOVA_FORCEUNROLL
                    for (int idr = 0; idr < ndim; ++idr) {
                        minus_one.u[idr] = loc_u_rec(i, j, k, 0, idim, idr);
                    }
                    minus_one.P = loc_P_rec(i, j, k, 0, idim);
                    EulerFlux const flux_L = compute_flux(minus_one, idim, m_eos);

                    EulerPrim plus_one; // Right, back, top
                    plus_one.rho = rho_rec(i, j, k, 1, idim);
                    NOVA_FORCEUNROLL
                    for (int idr = 0; idr < ndim; ++idr) {
                        plus_one.u[idr] = loc_u_rec(i, j, k, 1, idim, idr);
                    }
                    plus_one.P = loc_P_rec(i, j, k, 1, idim);
                    EulerFlux const flux_R = compute_flux(plus_one, idim, m_eos);

                    flux.rho += flux_L.rho * dsodv(0, idim) - flux_R.rho * dsodv(1, idim);
                    NOVA_FORCEUNROLL
                    for (int idr = 0; idr < ndim; ++idr) {
                        flux.rhou[idr] += flux_L.rhou[idr] * dsodv(0, idim) - flux_R.rhou[idr] * dsodv(1, idim);
                    }
                    flux.E += flux_L.E * dsodv(0, idim) - flux_R.E * dsodv(1, idim);

                    // Passive scalar
                    NOVA_FORCEUNROLL
                    for (int ifx = 0; ifx < nfx; ++ifx) {
                        double flux_fx_L = fx_rec(i, j, k, 0, idim, ifx) * flux_L.rho;
                        double flux_fx_R = fx_rec(i, j, k, 1, idim, ifx) * flux_R.rho;
                        flux_fx[ifx] += flux_fx_L * dsodv(0, idim) - flux_fx_R * dsodv(1, idim);
                    }
                }

                NOVA_FORCEUNROLL
                for (int iside = 0; iside < 2; ++iside) {
                    NOVA_FORCEUNROLL
                    for (int idim = 0; idim < ndim; ++idim) {
                        double Sg = 0;
                        NOVA_FORCEUNROLL
                        for (int idr = 0; idr < ndim; ++idr) {
                            Sg += m_gravity(i, j, k, idr, m_grid) * rhou_rec(i, j, k, iside, idim, idr);
                        }
                        E_rec(i, j, k, iside, idim) += dt * (flux.E + Sg);
                        NOVA_FORCEUNROLL
                        for (int idr = 0; idr < ndim; ++idr) {
                            rhou_rec(i, j, k, iside, idim, idr) += dt * (flux.rhou[idr] + m_gravity(i, j, k, idr, m_grid) * rho_rec(i, j, k, iside, idim));
                        }
                        rho_rec(i, j, k, iside, idim) += dt * flux.rho;
                        NOVA_FORCEUNROLL
                        for (int ifx = 0; ifx < nfx; ++ifx)
                        {
                            fx_rec(i, j, k, iside, idim, ifx) = (fx_rec(i, j, k, iside, idim, ifx) + dt * flux_fx[ifx]) / rho_rec(i, j, k, iside, idim);
                        }
                    }
                }
            });
    }
};

inline std::unique_ptr<IExtrapolationReconstruction> factory_time_reconstruction(
        std::string const& gravity,
        EOS const& eos,
        Grid const& grid,
        KV_double_1d g)
{
    if (gravity == "Uniform")
    {
        return std::make_unique<ExtrapolationTimeReconstruction<UniformGravity>>(UniformGravity(g), eos, grid);
    }
    throw std::runtime_error("Unknown time reconstruction algorithm: " + gravity + ".");
}

} // namespace novapp
