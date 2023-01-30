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
        Kokkos::View<const double***> rho_left,
        Kokkos::View<const double***> u_left,
        Kokkos::View<const double***> P_left,
        Kokkos::View<const double***> rho_right,
        Kokkos::View<const double***> u_right,
        Kokkos::View<const double***> P_right,
        Kokkos::View<double***> rhoex_left,
        Kokkos::View<double***> rhouex_left,
        Kokkos::View<double***> Eex_left,
        Kokkos::View<double***> rhoex_right,
        Kokkos::View<double***> rhouex_right,
        Kokkos::View<double***> Eex_right,
        thermodynamics::PerfectGas const& eos,
        double const dt, 
        double const dx) const 
        = 0;
};

class ExtrapolationCalculation : public IExtrapolationValues
{
public : 
    void execute(
        Kokkos::View<const double***> const rho_left,
        Kokkos::View<const double***> const u_left,
        Kokkos::View<const double***> const P_left,
        Kokkos::View<const double***> const rho_right,
        Kokkos::View<const double***> const u_right,
        Kokkos::View<const double***> const P_right,
        Kokkos::View<double***> const rhoex_left,
        Kokkos::View<double***> const rhouex_left,
        Kokkos::View<double***> const Eex_left,
        Kokkos::View<double***> const rhoex_right,
        Kokkos::View<double***> const rhouex_right,
        Kokkos::View<double***> const Eex_right,
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
        assert(rhoex_left.extent(0) == rhouex_left.extent(0));
        assert(rhouex_left.extent(0) == Eex_left.extent(0));
        assert(rhoex_left.extent(1) == rhouex_left.extent(1));
        assert(rhouex_left.extent(1) == Eex_left.extent(1));
        assert(rhoex_left.extent(2) == rhouex_left.extent(2));
        assert(rhouex_left.extent(2) == Eex_left.extent(2));
        assert(rhoex_right.extent(0) == rhouex_right.extent(0));
        assert(rhouex_right.extent(0) == Eex_right.extent(0));
        assert(rhoex_right.extent(1) == rhouex_right.extent(1));
        assert(rhouex_right.extent(1) == Eex_right.extent(1));
        assert(rhoex_right.extent(2) == rhouex_right.extent(2));
        assert(rhouex_right.extent(2) == Eex_right.extent(2));

        double dto2dx = dt / (2 * dx);

        Kokkos::parallel_for(
        "ExtrapolationCalculation",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
        {1, 0, 0},
        {static_cast<int>(rho_left.extent(0) - 1), 1, 1}),
        KOKKOS_LAMBDA(int i, int j, int k)
        {
            Flux left_flux(rho_left(i, j, k), u_left(i, j, k), P_left(i, j, k), eos);
            Flux right_flux(rho_right(i, j, k), u_right(i, j, k), P_right(i, j, k), eos);
            
            rhoex_left(i, j, k) =  rhoex_left(i, j, k) + dto2dx * (left_flux.FluxRho() - right_flux.FluxRho());
            rhouex_left(i, j, k) = rhouex_left(i, j, k) + dto2dx * (left_flux.FluxRhou() - right_flux.FluxRhou());
            Eex_left(i, j, k) = Eex_left(i, j, k) + dto2dx * (left_flux.FluxE() - right_flux.FluxE());
            rhoex_right(i, j, k) =  rhoex_right(i, j, k) + dto2dx * (left_flux.FluxRho() - right_flux.FluxRho());
            rhouex_right(i, j, k) = rhouex_right(i, j, k) + dto2dx * (left_flux.FluxRhou() - right_flux.FluxRhou());
            Eex_right(i, j, k) = Eex_right(i, j, k) + dto2dx * (left_flux.FluxE() - right_flux.FluxE());
        });   
    }
};