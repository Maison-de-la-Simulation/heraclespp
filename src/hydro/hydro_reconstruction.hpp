// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file hydro_reconstruction.hpp
//!

#pragma once

#include <memory>
#include <utility>

#include <Kokkos_Core.hpp>
#include <kokkos_shortcut.hpp>
#include <ndim.hpp>

#include "array_conversion.hpp"
#include "concepts.hpp"
#include "face_reconstruction.hpp"

namespace hclpp {

template <concepts::GravityField Gravity>
class IExtrapolationReconstruction;

class Grid;
class Range;

template <concepts::GravityField Gravity>
class IHydroReconstruction
{
public:
    IHydroReconstruction() = default;

    IHydroReconstruction(IHydroReconstruction const& rhs) = default;

    IHydroReconstruction(IHydroReconstruction&& rhs) noexcept = default;

    virtual ~IHydroReconstruction() noexcept = default;

    auto operator=(IHydroReconstruction const& rhs) -> IHydroReconstruction& = default;

    auto operator=(IHydroReconstruction&& rhs) noexcept -> IHydroReconstruction& = default;

    virtual void execute(
            Range const& range,
            Grid const& grid,
            Gravity const& gravity,
            double dt,
            KV_cdouble_3d const& rho,
            KV_cdouble_4d const& u,
            KV_cdouble_3d const& P,
            KV_cdouble_4d const& fx,
            KV_double_5d const& rho_rec,
            KV_double_6d const& rhou_rec,
            KV_double_5d const& E_rec,
            KV_double_6d const& fx_rec) const
            = 0;
};

template <concepts::EulerEoS EoS, concepts::GravityField Gravity>
class MUSCLHancockHydroReconstruction : public IHydroReconstruction<Gravity>
{
    std::unique_ptr<IFaceReconstruction> m_face_reconstruction;
    std::unique_ptr<IExtrapolationReconstruction<Gravity>> m_hancock_reconstruction;
    EoS m_eos;
    KV_double_5d m_p_rec;
    KV_double_6d m_u_rec;

public:
    MUSCLHancockHydroReconstruction(
            std::unique_ptr<IFaceReconstruction> face_reconstruction,
            std::unique_ptr<IExtrapolationReconstruction<Gravity>> hancock_reconstruction,
            EoS const& eos,
            KV_double_5d P_rec,
            KV_double_6d u_rec)
        : m_face_reconstruction(std::move(face_reconstruction))
        , m_hancock_reconstruction(std::move(hancock_reconstruction))
        , m_eos(eos)
        , m_p_rec(std::move(P_rec))
        , m_u_rec(std::move(u_rec))
    {
    }

    void execute(
            Range const& range,
            Grid const& grid,
            Gravity const& gravity,
            double const dt,
            KV_cdouble_3d const& rho,
            KV_cdouble_4d const& u,
            KV_cdouble_3d const& P,
            KV_cdouble_4d const& fx,
            KV_double_5d const& rho_rec,
            KV_double_6d const& rhou_rec,
            KV_double_5d const& E_rec,
            KV_double_6d const& fx_rec) const final
    {
        m_face_reconstruction->execute(range, grid, rho, rho_rec);
        for (int idim = 0; idim < ndim; ++idim) {
            m_face_reconstruction
                    ->execute(range, grid, Kokkos::subview(u, ALL, ALL, ALL, idim), Kokkos::subview(m_u_rec, ALL, ALL, ALL, ALL, ALL, idim));
        }
        m_face_reconstruction->execute(range, grid, P, m_p_rec);
        int const nfx = fx.extent_int(3);
        for (int ifx = 0; ifx < nfx; ++ifx) {
            m_face_reconstruction
                    ->execute(range, grid, Kokkos::subview(fx, ALL, ALL, ALL, ifx), Kokkos::subview(fx_rec, ALL, ALL, ALL, ALL, ALL, ifx));
        }

        for (int idim = 0; idim < ndim; ++idim) {
            for (int iside = 0; iside < 2; ++iside) {
                KV_cdouble_3d const rho_f = Kokkos::subview(rho_rec, ALL, ALL, ALL, iside, idim);
                Kokkos::Array<KV_cdouble_3d, ndim> u_f;
                for (int iv = 0; iv < ndim; ++iv) {
                    u_f[iv] = Kokkos::subview(m_u_rec, ALL, ALL, ALL, iside, idim, iv);
                }
                KV_cdouble_3d const P_f = Kokkos::subview(m_p_rec, ALL, ALL, ALL, iside, idim);
                Kokkos::Array<KV_double_3d, ndim> rhou_f;
                for (int iv = 0; iv < ndim; ++iv) {
                    rhou_f[iv] = Kokkos::subview(rhou_rec, ALL, ALL, ALL, iside, idim, iv);
                }
                KV_double_3d const E_f = Kokkos::subview(E_rec, ALL, ALL, ALL, iside, idim);
                conv_prim_to_cons(range, m_eos, rho_f, u_f, P_f, rhou_f, E_f);
            }
        }

        m_hancock_reconstruction->execute(range, grid, gravity, dt, m_u_rec, m_p_rec, rho_rec, rhou_rec, E_rec, fx_rec);
    }
};

} // namespace hclpp
