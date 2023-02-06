//!
//! @file godunov_scheme.hpp
//!
#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include <Kokkos_Core.hpp>
#include <PerfectGas.hpp>

#include "euler_equations.hpp"

class HLL
{
public:
    //! HLL solver
    //! @param[in] rhoL density left
    //! @param[in] uL speed left
    //! @param[in] PL pressure left
    //! @param[in] rhoR density right
    //! @param[in] uR speed right
    //! @param[in] PR pressure right
    //! @return intercell EulerFlux
    EulerFlux operator()(
            EulerCons const& consL,
            EulerCons const& consR,
            thermodynamics::PerfectGas const& eos) const noexcept
    {
        EulerPrim const primL = to_prim(consL, eos);
        EulerPrim const primR = to_prim(consR, eos);

        double const cL = eos.compute_speed_of_sound(primL.density, primL.pressure);
        double const cR = eos.compute_speed_of_sound(primR.density, primR.pressure);

        double const wsL = std::min(primL.velocity - cL, primR.velocity - cR);
        double const wsR = std::max(primL.velocity + cL, primR.velocity + cR);

        EulerFlux const fluxL = compute_flux(primL, eos);
        EulerFlux const fluxR = compute_flux(primR, eos);

        if (wsL >= 0)
        {
            return fluxL;
        }

        if (wsL <= 0 && wsR >= 0)
        {
            EulerFlux flux;
            flux.density
                    = FluxHLL(consL.density, consR.density, fluxL.density, fluxR.density, wsL, wsR);
            flux.momentum = FluxHLL(
                    consL.momentum,
                    consR.momentum,
                    fluxL.momentum,
                    fluxR.momentum,
                    wsL,
                    wsR);
            flux.energy = FluxHLL(consL.energy, consR.energy, fluxL.energy, fluxR.energy, wsL, wsR);
            return flux;
        }

        if (wsR <= 0)
        {
            return fluxR;
        }

        return EulerFlux {};
    }

    static double FluxHLL(
            double const UL,
            double const UR,
            double const FL,
            double const FR,
            double const wsL,
            double const wsR) noexcept
    {
        return (wsR * FL - wsL * FR + wsL * wsR * (UR - UL)) / (wsR - wsL);
    }
};

class IGodunovScheme
{
public:
    IGodunovScheme() = default;

    IGodunovScheme(IGodunovScheme const& x) = default;

    IGodunovScheme(IGodunovScheme&& x) noexcept = default;

    virtual ~IGodunovScheme() noexcept = default;

    IGodunovScheme& operator=(IGodunovScheme const& x) = default;

    IGodunovScheme& operator=(IGodunovScheme&& x) noexcept = default;

    virtual void execute(
            Kokkos::View<const double***> density,
            Kokkos::View<const double***> momentum,
            Kokkos::View<const double***> energy,
            Kokkos::View<const double***> densityL,
            Kokkos::View<const double***> momentumL,
            Kokkos::View<const double***> energyL,
            Kokkos::View<const double***> densityR,
            Kokkos::View<const double***> momentumR,
            Kokkos::View<const double***> energyR,
            Kokkos::View<double***> density_new,
            Kokkos::View<double***> momentum_new,
            Kokkos::View<double***> energy_new,
            double dt) const
            = 0;
};

template <class RiemannSolver>
class RiemannBasedGodunovScheme : public IGodunovScheme
{
    static_assert(
            std::is_invocable_r_v<
                    EulerFlux,
                    RiemannSolver,
                    EulerCons,
                    EulerCons,
                    thermodynamics::PerfectGas>,
            "Incompatible Riemann solver.");

    RiemannSolver m_riemann_solver;

    thermodynamics::PerfectGas m_eos;

    double m_dx;

public:
    RiemannBasedGodunovScheme(
            RiemannSolver const& riemann_solver,
            thermodynamics::PerfectGas const& eos,
            double const dx)
        : m_riemann_solver(riemann_solver)
        , m_eos(eos)
        , m_dx(dx)
    {
    }

    void execute(
            Kokkos::View<const double***> const density,
            Kokkos::View<const double***> const momentum,
            Kokkos::View<const double***> const energy,
            Kokkos::View<const double***> const densityL,
            Kokkos::View<const double***> const momentumL,
            Kokkos::View<const double***> const energyL,
            Kokkos::View<const double***> const densityR,
            Kokkos::View<const double***> const momentumR,
            Kokkos::View<const double***> const energyR,
            Kokkos::View<double***> const density_new,
            Kokkos::View<double***> const momentum_new,
            Kokkos::View<double***> const energy_new,
            double dt) const final
    {
        double dtodx = dt / m_dx;
        Kokkos::parallel_for(
                "RiemannBasedGodunovScheme",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                        {2, 0, 0},
                        {density.extent(0) - 2, density.extent(1), density.extent(2)}),
                KOKKOS_LAMBDA(int i, int j, int k) {
                    EulerCons consR_im1jk;
                    consR_im1jk.density = densityR(i - 1, j, k);
                    consR_im1jk.momentum = momentumR(i - 1, j, k);
                    consR_im1jk.energy = energyR(i - 1, j, k);
                    EulerCons consL_ijk;
                    consL_ijk.density = densityL(i, j, k);
                    consL_ijk.momentum = momentumL(i, j, k);
                    consL_ijk.energy = energyL(i, j, k);
                    EulerFlux const FluxL = m_riemann_solver(consR_im1jk, consL_ijk, m_eos);

                    EulerCons consR_ijk;
                    consR_ijk.density = densityR(i, j, k);
                    consR_ijk.momentum = momentumR(i, j, k);
                    consR_ijk.energy = energyR(i, j, k);
                    EulerCons consL_ip1jk;
                    consL_ip1jk.density = densityL(i + 1, j, k);
                    consL_ip1jk.momentum = momentumL(i + 1, j, k);
                    consL_ip1jk.energy = energyL(i + 1, j, k);
                    EulerFlux const FluxR = m_riemann_solver(consR_ijk, consL_ip1jk, m_eos);

                    density_new(i, j, k)
                            = density(i, j, k) + dtodx * (FluxL.density - FluxR.density);
                    momentum_new(i, j, k)
                            = momentum(i, j, k) + dtodx * (FluxL.momentum - FluxR.momentum);
                    energy_new(i, j, k) = energy(i, j, k) + dtodx * (FluxL.energy - FluxR.energy);
                });
    }
};

inline std::unique_ptr<IGodunovScheme> factory_godunov_scheme(
        std::string const& riemann_solver,
        thermodynamics::PerfectGas const& eos,
        double const dx)
{
    if (riemann_solver == "HLL")
    {
        return std::make_unique<RiemannBasedGodunovScheme<HLL>>(HLL(), eos, dx);
    }

    throw std::runtime_error("Invalid riemann solver: " + riemann_solver + ".");
}
