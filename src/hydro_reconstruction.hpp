//!
//! @file hydro_reconstruction.hpp
//!

#pragma once

#include "Kokkos_shortcut.hpp"
#include "array_conversion.hpp"
#include "euler_equations.hpp"
#include "face_reconstruction.hpp"
#include "ndim.hpp"
#include "range.hpp"

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
            KV_double_5d rho_rec,
            KV_double_6d rhou_rec,
            KV_double_5d E_rec,
            KV_cdouble_3d rho,
            KV_cdouble_4d u,
            KV_cdouble_3d P,
            thermodynamics::PerfectGas const& eos,
            KV_cdouble_1d dx,
            double const dt) const
            = 0;
};

class MUSCLHancockHydroReconstruction : public IHydroReconstruction
{
    // WARNING, needs to be shared to be captured by KOKKOS_CLASS_LAMBDA
    std::shared_ptr<IFaceReconstruction> m_face_reconstruction;

    KV_double_5d m_P_rec;

    KV_double_6d m_u_rec;

public:
    MUSCLHancockHydroReconstruction(
            std::unique_ptr<IFaceReconstruction> face_reconstruction,
            KV_double_5d P_rec,
            KV_double_6d u_rec)
        : m_face_reconstruction(std::move(face_reconstruction))
        , m_P_rec(P_rec)
        , m_u_rec(u_rec)
    {
    }

    void execute(
            Range const& range,
            KV_double_5d const rho_rec,
            KV_double_6d const rhou_rec,
            KV_double_5d const E_rec,
            KV_cdouble_3d const rho,
            KV_cdouble_4d const u,
            KV_cdouble_3d const P,
            thermodynamics::PerfectGas const& eos,
            KV_cdouble_1d dx,
            double const dt) const final
    {
        m_face_reconstruction->execute(range, rho, rho_rec, dx);
        m_face_reconstruction->execute(range, P, m_P_rec, dx);
        for (int idim = 0; idim < ndim; ++idim)
        {
            m_face_reconstruction->execute(
                    range,
                    Kokkos::subview(u, ALL, ALL, ALL, idim),
                    Kokkos::subview(m_u_rec, ALL, ALL, ALL, ALL, ALL, idim),
                    dx);
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
                        eos);
            }
        }

        auto const [begin, end] = cell_range(range);
        Kokkos::parallel_for(
            "HancockExtrapolation",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
            KOKKOS_CLASS_LAMBDA(int i, int j, int k) {
                for (int idim = 0; idim < ndim; ++idim)
                {
                    EulerPrim minus_one; // Left, front, bottom
                    minus_one.density = rho_rec(i, j, k, 0, idim);
                    for (int idr = 0; idr < ndim; ++idr)
                    {
                        minus_one.velocity[idr] = m_u_rec(i, j, k, 0, idim, idr);
                    }
                    minus_one.pressure = m_P_rec(i, j, k, 0, idim);
                    EulerFlux flux_minus_one = compute_flux(minus_one, idim, eos);

                    EulerPrim plus_one; // Right, back, top
                    plus_one.density = rho_rec(i, j, k, 1, idim);
                    for (int idr = 0; idr < ndim; ++idr)
                    {
                        plus_one.velocity[idr] = m_u_rec(i, j, k, 1, idim, idr);
                    }
                    plus_one.pressure = m_P_rec(i, j, k, 1, idim);
                    EulerFlux flux_plus_one = compute_flux(plus_one, idim, eos);

                    double dto2dx = dt / (2 * dx(idim));

                    for (int ipos = 0; ipos < ndim; ++ipos)
                    {
                        rho_rec(i, j, k, 0, ipos) += dto2dx * (flux_minus_one.density - flux_plus_one.density);
                        rho_rec(i, j, k, 1, ipos) += dto2dx * (flux_minus_one.density - flux_plus_one.density);
                        for (int idr = 0; idr < ndim; ++idr)
                        {
                            rhou_rec(i, j, k, 0, ipos, idr) += dto2dx * (flux_minus_one.momentum[idr] - flux_plus_one.momentum[idr]);
                            rhou_rec(i, j, k, 1, ipos, idr) += dto2dx * (flux_minus_one.momentum[idr] - flux_plus_one.momentum[idr]);
                        }
                        E_rec(i, j, k, 0, ipos) += dto2dx * (flux_minus_one.energy - flux_plus_one.energy);
                        E_rec(i, j, k, 1, ipos) += dto2dx * (flux_minus_one.energy - flux_plus_one.energy);
                    }
                }
            });
    }
};

} // namespace novapp
