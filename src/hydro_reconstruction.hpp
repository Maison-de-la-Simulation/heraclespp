//!
//! @file hydro_reconstruction.hpp
//!

#pragma once

#include "Kokkos_shortcut.hpp"
#include "array_conversion.hpp"
#include "face_reconstruction.hpp"
#include "extrapolation_time.hpp"
#include "ndim.hpp"
#include "range.hpp"
#include "grid.hpp"
#include "nova_params.hpp"

namespace novapp
{

class IHydroReconstruction
{
public:
    IHydroReconstruction() = default;

    IHydroReconstruction(IHydroReconstruction const& x) = default;

    IHydroReconstruction(IHydroReconstruction&& x) noexcept = default;

    virtual ~IHydroReconstruction() noexcept = default;

    IHydroReconstruction& operator=(IHydroReconstruction const& x) = default;

    IHydroReconstruction& operator=(IHydroReconstruction&& x) noexcept = default;

    virtual void execute(
            Range const& range,
            double const dt,
            KV_double_5d rho_rec,
            KV_double_6d rhou_rec,
            KV_double_5d E_rec,
            KV_double_6d fx_rec,
            KV_cdouble_3d rho,
            KV_cdouble_4d u,
            KV_cdouble_3d P,
            KV_cdouble_4d fx) const
            = 0;
};

class MUSCLHancockHydroReconstruction : public IHydroReconstruction
{
    // WARNING, needs to be shared to be captured by KOKKOS_CLASS_LAMBDA
    std::shared_ptr<IFaceReconstruction> m_face_reconstruction;

    std::shared_ptr<IExtrapolationReconstruction> m_hancock_reconstruction;

    thermodynamics::PerfectGas m_eos;

    KV_double_5d m_P_rec;

    KV_double_6d m_u_rec;

public:
    MUSCLHancockHydroReconstruction(
            std::unique_ptr<IFaceReconstruction> face_reconstruction,
            std::unique_ptr<IExtrapolationReconstruction> hancock_reconstruction,
            thermodynamics::PerfectGas const& eos,
            KV_double_5d P_rec,
            KV_double_6d u_rec)
        : m_face_reconstruction(std::move(face_reconstruction))
        , m_hancock_reconstruction(std::move(hancock_reconstruction))
        , m_eos(eos)
        , m_P_rec(P_rec)
        , m_u_rec(u_rec)
    {
    }

    void execute(
            Range const& range,
            double const dt,
            KV_double_5d const rho_rec,
            KV_double_6d const rhou_rec,
            KV_double_5d const E_rec,
            KV_double_6d const fx_rec,
            KV_cdouble_3d const rho,
            KV_cdouble_4d const u,
            KV_cdouble_3d const P,
            KV_cdouble_4d const fx) const final
    {
        m_face_reconstruction->execute(range, rho, rho_rec);
        m_face_reconstruction->execute(range, P, m_P_rec);
        for (int idim = 0; idim < ndim; ++idim)
        {
            m_face_reconstruction->execute(
                    range,
                    Kokkos::subview(u, ALL, ALL, ALL, idim),
                    Kokkos::subview(m_u_rec, ALL, ALL, ALL, ALL, ALL, idim));
        }
        int nfx = fx.extent_int(3);
        for (int ifx = 0; ifx < nfx; ++ifx)
        {
            m_face_reconstruction->execute(
                    range,
                    Kokkos::subview(fx, ALL, ALL, ALL, ifx),
                    Kokkos::subview(fx_rec, ALL, ALL, ALL, ALL, ALL, ifx));
        }

        for (int idim = 0; idim < ndim; ++idim)
        {
            for (int iside = 0; iside < 2; ++iside)
            {
                conv_prim_to_cons(
                        range,
                        Kokkos::subview(rhou_rec, ALL, ALL, ALL, iside, idim, ALL),
                        Kokkos::subview(E_rec, ALL, ALL, ALL, iside, idim),
                        Kokkos::subview(rho_rec, ALL, ALL, ALL, iside, idim),
                        Kokkos::subview(m_u_rec, ALL, ALL, ALL, iside, idim, ALL),
                        Kokkos::subview(m_P_rec, ALL, ALL, ALL, iside, idim),
                        m_eos);
            }
        }

        m_hancock_reconstruction->execute(range, dt, rho_rec, rhou_rec, E_rec, m_u_rec, m_P_rec, fx_rec);
    }
};

} // namespace novapp