//!
//! @file extrapolation_construction.hpp
//!
#pragma once

#include "euler_equations.hpp"

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
        Kokkos::View<double******> const rhou,
        Kokkos::View<double*****> const E,
        Kokkos::View<double*****> const rho,
        Kokkos::View<const double******> const u,
        Kokkos::View<const double*****> const P,
        thermodynamics::PerfectGas const& eos,
        Kokkos::View<const double*> dx,
        double const dt) const 
        = 0;
};

class ExtrapolationCalculation : public IExtrapolationValues
{
public : 
    void execute(
        Kokkos::View<double******> const rhou,
        Kokkos::View<double*****> const E,
        Kokkos::View<double*****> const rho,
        Kokkos::View<const double******> const u,
        Kokkos::View<const double*****> const P,
        thermodynamics::PerfectGas const& eos,
        Kokkos::View<const double*> dx,
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

        int istart = 1; // Default = 1D
        int jstart = 0;
        int kstart = 0;
        int iend = rho.extent(0) - 1;
        int jend = 1;
        int kend = 1;
            
        if (ndim == 2) // 2D
        {
            jstart = 1;
            jend = rho.extent(1) - 1;
        }
        if (ndim == 3) // 3D
        {
            jstart = 1;
            kstart = 1;
            jend = rho.extent(1) - 1;
            kend = rho.extent(2) - 1;
        }

        Kokkos::parallel_for(
        "ExtrapolationCalculation",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
        {istart, jstart, kstart},
        {iend, jend, kend}),
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
                rho(i, j, k, 0, idim) += dto2dx * (flux_minus_one.density - flux_plus_one.density);
                rho(i, j, k, 1, idim) += dto2dx * (flux_minus_one.density - flux_plus_one.density);
                for (int idr = 0; idr < ndim; ++idr)
                {
                    rhou(i, j, k, 0, idim, idr) += dto2dx * (flux_minus_one.momentum[idr] - flux_plus_one.momentum[idr]);
                    rhou(i, j, k, 1, idim, idr) += dto2dx * (flux_minus_one.momentum[idr] - flux_plus_one.momentum[idr]);
                }
                E(i, j, k, 0, idim) += dto2dx * (flux_minus_one.energy - flux_plus_one.energy);
                E(i, j, k, 1, idim) += dto2dx * (flux_minus_one.energy - flux_plus_one.energy);
            }
        });   
    }
};
