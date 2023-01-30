//!
//! @file extrapolation_construction.hpp
//!
#pragma once

#include "flux_definition.hpp"

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
        KOKKOS_LAMBDA(int i, int j, int k)
        {
            Flux left_flux(rho_left(i, j, k), u_left(i, j, k), P_left(i, j, k), eos);
            Flux right_flux(rho_right(i, j, k), u_right(i, j, k), P_right(i, j, k), eos);
            
            rho_left(i, j, k) += dto2dx * (left_flux.FluxRho() - right_flux.FluxRho());
            rhou_left(i, j, k) += dto2dx * (left_flux.FluxRhou() - right_flux.FluxRhou());
            E_left(i, j, k) += dto2dx * (left_flux.FluxE() - right_flux.FluxE());
            rho_right(i, j, k) += dto2dx * (left_flux.FluxRho() - right_flux.FluxRho());
            rhou_right(i, j, k) += dto2dx * (left_flux.FluxRhou() - right_flux.FluxRhou());
            E_right(i, j, k) += dto2dx * (left_flux.FluxE() - right_flux.FluxE());
        });   
    }
};