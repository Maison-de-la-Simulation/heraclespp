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
#include "extrapolation_time.hpp"
#include "face_reconstruction.hpp"

namespace novapp
{

class Grid;
class Range;

template <class Gravity>
class IHydroReconstruction
{
public:
    IHydroReconstruction() = default;

    IHydroReconstruction(IHydroReconstruction const& rhs) = default;

    IHydroReconstruction(IHydroReconstruction&& rhs) noexcept = default;

    virtual ~IHydroReconstruction() noexcept = default;

    IHydroReconstruction& operator=(IHydroReconstruction const& rhs) = default;

    IHydroReconstruction& operator=(IHydroReconstruction&& rhs) noexcept = default;

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

template <class EoS, class Gravity>
class MUSCLHancockHydroReconstruction : public IHydroReconstruction<Gravity>
{
    std::unique_ptr<IFaceReconstruction> m_face_reconstruction;
    std::unique_ptr<IExtrapolationReconstruction<Gravity>> m_hancock_reconstruction;
    EoS m_eos;
    KV_double_5d m_P_rec;
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
        , m_P_rec(std::move(P_rec))
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
        for (int idim = 0; idim < ndim; ++idim)
        {
            m_face_reconstruction->execute(
                    range,
                    grid,
                    Kokkos::subview(u, ALL, ALL, ALL, idim),
                    Kokkos::subview(m_u_rec, ALL, ALL, ALL, ALL, ALL, idim));
        }
        m_face_reconstruction->execute(range, grid, P, m_P_rec);
        int const nfx = fx.extent_int(3);
        for (int ifx = 0; ifx < nfx; ++ifx)
        {
            m_face_reconstruction->execute(
                    range,
                    grid,
                    Kokkos::subview(fx, ALL, ALL, ALL, ifx),
                    Kokkos::subview(fx_rec, ALL, ALL, ALL, ALL, ALL, ifx));
        }

        for (int idim = 0; idim < ndim; ++idim)
        {
            for (int iside = 0; iside < 2; ++iside)
            {
                conv_prim_to_cons(
                        range,
                        m_eos,
                        Kokkos::subview(rho_rec, ALL, ALL, ALL, iside, idim),
                        Kokkos::subview(m_u_rec, ALL, ALL, ALL, iside, idim, ALL),
                        Kokkos::subview(m_P_rec, ALL, ALL, ALL, iside, idim),
                        Kokkos::subview(rhou_rec, ALL, ALL, ALL, iside, idim, ALL),
                        Kokkos::subview(E_rec, ALL, ALL, ALL, iside, idim));
            }
        }

        m_hancock_reconstruction->execute(range, grid, gravity, dt, m_u_rec, m_P_rec, rho_rec, rhou_rec, E_rec, fx_rec);
    }
};

} // namespace novapp
