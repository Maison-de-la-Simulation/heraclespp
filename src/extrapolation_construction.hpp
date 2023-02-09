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
        Kokkos::View<double***> rho_left,
        Kokkos::View<const double***> u_left,
        Kokkos::View<const double***> P_left,
        Kokkos::View<double***> rho_right,
        Kokkos::View<const double***> u_right,
        Kokkos::View<const double***> P_right,
        Kokkos::View<double***> rhou_left,
        Kokkos::View<double***> E_left,
        Kokkos::View<double***> rhou_right,
        Kokkos::View<double***> E_right,
        thermodynamics::PerfectGas const& eos,
        double const dt, 
        double const dx) const 
        = 0;
};

class ExtrapolationCalculation : public IExtrapolationValues
{
public : 
    void execute(
        Kokkos::View<double***> const rho_left,
        Kokkos::View<const double***> const u_left,
        Kokkos::View<const double***> const P_left,
        Kokkos::View<double***> const rho_right,
        Kokkos::View<const double***> const u_right,
        Kokkos::View<const double***> const P_right,
        Kokkos::View<double***> const rhou_left,
        Kokkos::View<double***> const E_left,
        Kokkos::View<double***> const rhou_right,
        Kokkos::View<double***> const E_right,
        thermodynamics::PerfectGas const& eos,
        double const dt, 
        double const dx) const final
    {
        assert(rho_left.extent(0) == u_left.extent(0));
        assert(u_left.extent(0) == P_left.extent(0));
        assert(rho_left.extent(1) == u_left.extent(1));
        assert(u_left.extent(1) == P_left.extent(1));
        assert(rho_left.extent(2) == u_left.extent(2));
        assert(u_left.extent(2) == P_left.extent(2));
        assert(rho_right.extent(0) == u_right.extent(0));
        assert(u_right.extent(0) == P_right.extent(0));
        assert(rho_right.extent(1) == u_right.extent(1));
        assert(u_right.extent(1) == P_right.extent(1));
        assert(rho_right.extent(2) == u_right.extent(2));
        assert(u_right.extent(2) == P_right.extent(2));
        assert(rhou_left.extent(0) == E_left.extent(0));
        assert(rhou_left.extent(1) == E_left.extent(1));
        assert(rhou_left.extent(2) == E_left.extent(2));
        assert(rhou_right.extent(0) == E_right.extent(0));
        assert(rhou_right.extent(1) == E_right.extent(1));
        assert(rhou_right.extent(2) == E_right.extent(2));

        double dto2dx = dt / (2 * dx);

        Kokkos::parallel_for(
        "ExtrapolationCalculation",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
        {1, 0, 0},
        {rho_left.extent(0) - 1, rho_left.extent(1), rho_left.extent(2)}),
        KOKKOS_CLASS_LAMBDA(int i, int j, int k)
        {
            EulerPrim prim_left;
            prim_left.density = rho_left(i, j, k);
            prim_left.velocity = u_left(i, j, k);
            prim_left.pressure = P_left(i, j, k);
            EulerFlux const flux_left = compute_flux(prim_left, eos);
            EulerPrim prim_right;
            prim_right.density = rho_right(i, j, k);
            prim_right.velocity = u_right(i, j, k);
            prim_right.pressure = P_right(i, j, k);
            EulerFlux const flux_right = compute_flux(prim_right, eos);
            
            rho_left(i, j, k) += dto2dx * (flux_left.density - flux_right.density);
            rhou_left(i, j, k) += dto2dx * (flux_left.momentum - flux_right.momentum);
            E_left(i, j, k) += dto2dx * (flux_left.energy - flux_right.energy);
            rho_right(i, j, k) += dto2dx * (flux_left.density - flux_right.density);
            rhou_right(i, j, k) += dto2dx * (flux_left.momentum - flux_right.momentum);
            E_right(i, j, k) += dto2dx * (flux_left.energy - flux_right.energy);
        });   
    }
};
