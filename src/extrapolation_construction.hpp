//!
//! @file extrapolation_construction.hpp
//!

#pragma once

#include "Kokkos_shortcut.hpp"
#include "euler_equations.hpp"
#include "ndim.hpp"
#include "range.hpp"

namespace novapp
{

class IExtrapolationValues
{
public:
    IExtrapolationValues() = default;

    IExtrapolationValues(IExtrapolationValues const& x) = default;

    IExtrapolationValues(IExtrapolationValues&& x) noexcept = default;

    virtual ~IExtrapolationValues() noexcept = default;

    IExtrapolationValues& operator=(IExtrapolationValues const& x) = default;

    IExtrapolationValues& operator=(IExtrapolationValues&& x) noexcept = default;

    virtual void execute(
        Range const& range,
        KV_double_6d const rhou,
        KV_double_5d const E,
        KV_double_5d const rho,
        KV_cdouble_6d const u,
        KV_cdouble_5d const P,
        thermodynamics::PerfectGas const& eos,
        KV_cdouble_1d dx,
        double const dt) const 
        = 0;
};

class ExtrapolationCalculation : public IExtrapolationValues
{
public : 
    void execute(
        Range const& range,
        KV_double_6d const rhou,
        KV_double_5d const E,
        KV_double_5d const rho,
        KV_cdouble_6d const u,
        KV_cdouble_5d const P,
        thermodynamics::PerfectGas const& eos,
        KV_cdouble_1d dx,
        double const dt) const final
    {
        assert(rhou.extent(0) == E.extent(0));
        assert(E.extent(0) == rho.extent(0));
        assert(rho.extent(0) == u.extent(0));
        assert(u.extent(0) == P.extent(0));
        assert(rhou.extent(1) == E.extent(1));
        assert(E.extent(1) == rho.extent(1));
        assert(rho.extent(1) == u.extent(1));
        assert(u.extent(1) == P.extent(1));
        assert(rhou.extent(2) == E.extent(2));
        assert(E.extent(2) == rho.extent(2));
        assert(rho.extent(2) == u.extent(2));
        assert(u.extent(2) == P.extent(2));
        assert(rhou.extent(3) == E.extent(3));
        assert(E.extent(3) == rho.extent(3));
        assert(rho.extent(3) == u.extent(3));
        assert(u.extent(3) == P.extent(3));
        assert(rhou.extent(4) == E.extent(4));
        assert(E.extent(4) == rho.extent(4));
        assert(rho.extent(4) == u.extent(4));
        assert(u.extent(4) == P.extent(4));
        assert(rhou.extent(5) == u.extent(5));

        auto const [begin, end] = cell_range(range);
        Kokkos::parallel_for(
        "ExtrapolationCalculation",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
        KOKKOS_CLASS_LAMBDA(int i, int j, int k)
        {
            for (int idim = 0; idim < ndim; ++idim)
            {
                EulerPrim minus_one; // Left, front, bottom
                minus_one.density = rho(i, j, k, 0, idim);
                for (int idr = 0; idr < ndim; ++idr)
                {
                    minus_one.velocity[idr] = u(i, j, k, 0, idim, idr);
                }
                minus_one.pressure = P(i, j, k, 0, idim);
                EulerFlux flux_minus_one = compute_flux(minus_one, idim, eos);

                EulerPrim plus_one; // Right, back, top
                plus_one.density = rho(i, j, k, 1, idim);
                for (int idr = 0; idr < ndim; ++idr)
                {
                    plus_one.velocity[idr] = u(i, j, k, 1, idim, idr);
                }
                plus_one.pressure = P(i, j, k, 1, idim);
                EulerFlux flux_plus_one = compute_flux(plus_one, idim, eos);

                double dto2dx = dt / (2 * dx(idim));

                for (int ipos = 0; ipos < ndim; ++ipos)
                {
                    rho(i, j, k, 0, ipos) += dto2dx * (flux_minus_one.density - flux_plus_one.density);
                    rho(i, j, k, 1, ipos) += dto2dx * (flux_minus_one.density - flux_plus_one.density);
                    for (int idr = 0; idr < ndim; ++idr)
                    {
                        rhou(i, j, k, 0, ipos, idr) += dto2dx * (flux_minus_one.momentum[idr] - flux_plus_one.momentum[idr]);
                        rhou(i, j, k, 1, ipos, idr) += dto2dx * (flux_minus_one.momentum[idr] - flux_plus_one.momentum[idr]);
                    }
                    E(i, j, k, 0, ipos) += dto2dx * (flux_minus_one.energy - flux_plus_one.energy);
                    E(i, j, k, 1, ipos) += dto2dx * (flux_minus_one.energy - flux_plus_one.energy);
                }
            }
        });   
    }
};

} // namespace novapp
