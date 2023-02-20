//!
//! @file initialisation_problem.hpp
//!
#pragma once

#include <Kokkos_Core.hpp>

#include "euler_equations.hpp"

class IInitialisationProblem
{
public:
    IInitialisationProblem() = default;

    IInitialisationProblem(IInitialisationProblem const& x) = default;

    IInitialisationProblem(IInitialisationProblem&& x) noexcept = default;

    virtual ~IInitialisationProblem() noexcept = default;

    IInitialisationProblem& operator=(IInitialisationProblem const& x) = default;

    IInitialisationProblem& operator=(IInitialisationProblem&& x) noexcept = default;

    virtual void execute(
        Kokkos::View<double***> rho,
        Kokkos::View<double****> u,
        Kokkos::View<double***> P,
        Kokkos::View<const double*> nodes_x0) const 
        = 0;
};

class ShockTube : public IInitialisationProblem
{
public:
    void execute(
        Kokkos::View<double***> const rho,
        Kokkos::View<double****> const u,
        Kokkos::View<double***> const P,
        Kokkos::View<const double*> const nodes_x0) const final
    {
        assert(rho.extent(0) == u.extent(0));
        assert(u.extent(0) == P.extent(0));
        assert(rho.extent(1) == u.extent(1));
        assert(u.extent(1) == P.extent(1));
        assert(rho.extent(2) == u.extent(2));
        assert(u.extent(2) == P.extent(2));

        Kokkos::parallel_for(
        "ShockTubeInit",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
        {0, 0, 0},
        {rho.extent(0), rho.extent(1), rho.extent(2)}),
        KOKKOS_CLASS_LAMBDA(int i, int j, int k)
        {
            if((nodes_x0(i) + nodes_x0(i+1)) / 2 <= 0.5)
            {
                rho(i, j, k) = 1;
                P(i, j, k) = 1;
                for (int idim = 0; idim < ndim; ++idim)
                {
                    u(i, j, k, idim) = 0;
                }
            }
            else
            {
                rho(i, j, k) = 0.125;
                P(i, j, k) =  0.1;
                for (int idim = 0; idim < ndim; ++idim)
                {
                    u(i, j, k, idim) = 0;
                }
            }
        });  
    }
};

class AdvectionSinus : public IInitialisationProblem
{
public:
    void execute(
        Kokkos::View<double***> rho,
        Kokkos::View<double****> u,
        Kokkos::View<double***> P,
        Kokkos::View<const double*> const nodes_x0) const final
    {
        assert(rho.extent(0) == u.extent(0));
        assert(u.extent(0) == P.extent(0));
        assert(rho.extent(1) == u.extent(1));
        assert(u.extent(1) == P.extent(1));
        assert(rho.extent(2) == u.extent(2));
        assert(u.extent(2) == P.extent(2));

        Kokkos::parallel_for(
        "AdvectionInitSinus",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
        {0, 0, 0},
        {rho.extent(0), rho.extent(1), rho.extent(2)}),
        KOKKOS_CLASS_LAMBDA(int i, int j, int k)
        {
            rho(i, j, k) = 1 * std::exp(-15*std::pow(1./2 - (nodes_x0(i)+nodes_x0(i+1))/2, 2));
            P(i, j, k) = 0.1;
            for (int idim = 0; idim < ndim; ++idim)
            {
                u(i, j, k, idim) = 1;
            }
        });
    }
};

class AdvectionCrenel : public IInitialisationProblem
{
public:
    void execute(
        Kokkos::View<double***> rho,
        Kokkos::View<double****> u,
        Kokkos::View<double***> P,
        Kokkos::View<const double*> const nodes_x0) const final
    {
        assert(rho.extent(0) == u.extent(0));
        assert(u.extent(0) == P.extent(0));
        assert(rho.extent(1) == u.extent(1));
        assert(u.extent(1) == P.extent(1));
        assert(rho.extent(2) == u.extent(2));
        assert(u.extent(2) == P.extent(2));

        Kokkos::parallel_for(
        "AdvectionInitCrenel",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
        {0, 0, 0},
        {rho.extent(0), rho.extent(1), rho.extent(2)}),
        KOKKOS_CLASS_LAMBDA(int i, int j, int k)
        {
            if ( (nodes_x0(i) + nodes_x0(i+1)) / 2 <= 0.3)
            {
                rho(i, j, k) = 1;
            }
            else if ( (nodes_x0(i) + nodes_x0(i+1)) / 2 >= 0.7)
            {
                rho(i, j, k) = 1;
            }
            else
            {
                rho(i, j, k) = 2;
            }
            P(i, j, k) = 0.1;
            for (int idim = 0; idim < ndim; ++idim)
            {
                u(i, j, k, idim) = 1;
            }
        });
    }
};

inline std::unique_ptr<IInitialisationProblem> factory_initialisation(
    std::string const& s)
{
    if (s == "ShockTube")
    {
        return std::make_unique<ShockTube>();
    }
    if (s == "AdvectionSinus")
    {
        return std::make_unique<AdvectionSinus>();
    }
    if (s == "AdvectionCrenel")
    {
        return std::make_unique<AdvectionCrenel>();
    }
    throw std::runtime_error("Unknown boundary condition : " + s + ".");
}
