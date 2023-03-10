//!
//! @file initialisation_problem.hpp
//!
#pragma once

#include <Kokkos_Core.hpp>
#include <PerfectGas.hpp>
#include <math.h>

#include "euler_equations.hpp"
#include "ndim.hpp"
#include "range.hpp"

namespace novapp
{

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
        Range const& range,
        Kokkos::View<double***> rho,
        Kokkos::View<double****> u,
        Kokkos::View<double***> P,
        Kokkos::View<const double*> nodes_x0,
        [[maybe_unused]] Kokkos::View<const double*> nodes_y0,
        [[maybe_unused]]Kokkos::View<double*> g_array) const 
        = 0;
};

class ShockTube : public IInitialisationProblem
{
public:
    void execute(
        Range const& range,
        Kokkos::View<double***> const rho,
        Kokkos::View<double****> const u,
        Kokkos::View<double***> const P,
        Kokkos::View<const double*> const nodes_x0,
        [[maybe_unused]] Kokkos::View<const double*> const nodes_y0,
        [[maybe_unused]]Kokkos::View<double*> g_array) const final
    {
        assert(rho.extent(0) == u.extent(0));
        assert(u.extent(0) == P.extent(0));
        assert(rho.extent(1) == u.extent(1));
        assert(u.extent(1) == P.extent(1));
        assert(rho.extent(2) == u.extent(2));
        assert(u.extent(2) == P.extent(2));

        auto const [begin, end] = cell_range(range);
        Kokkos::parallel_for(
        "ShockTubeInit",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
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
        Range const& range,
        Kokkos::View<double***> rho,
        Kokkos::View<double****> u,
        Kokkos::View<double***> P,
        Kokkos::View<const double*> const nodes_x0,
        [[maybe_unused]] Kokkos::View<const double*> const nodes_y0,
        [[maybe_unused]]Kokkos::View<double*> g_array) const final
    {
        assert(rho.extent(0) == u.extent(0));
        assert(u.extent(0) == P.extent(0));
        assert(rho.extent(1) == u.extent(1));
        assert(u.extent(1) == P.extent(1));
        assert(rho.extent(2) == u.extent(2));
        assert(u.extent(2) == P.extent(2));

        auto const [begin, end] = cell_range(range);
        Kokkos::parallel_for(
        "AdvectionInitSinus",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
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
        Range const& range,
        Kokkos::View<double***> rho,
        Kokkos::View<double****> u,
        Kokkos::View<double***> P,
        Kokkos::View<const double*> const nodes_x0,
        [[maybe_unused]] Kokkos::View<const double*> const nodes_y0,
        [[maybe_unused]]Kokkos::View<double*> g_array) const final
    {
        assert(rho.extent(0) == u.extent(0));
        assert(u.extent(0) == P.extent(0));
        assert(rho.extent(1) == u.extent(1));
        assert(u.extent(1) == P.extent(1));
        assert(rho.extent(2) == u.extent(2));
        assert(u.extent(2) == P.extent(2));

        auto const [begin, end] = cell_range(range);
        Kokkos::parallel_for(
        "AdvectionInitCrenel",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
        KOKKOS_CLASS_LAMBDA(int i, int j, int k)
        {
            if ((nodes_x0(i) + nodes_x0(i+1)) / 2 <= 0.3)
            {
                rho(i, j, k) = 1;
            }
            else if ((nodes_x0(i) + nodes_x0(i+1)) / 2 >= 0.7)
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

class GreshoVortex : public IInitialisationProblem
{
public:
    void execute(
        Range const& range,
        Kokkos::View<double***> const rho,
        Kokkos::View<double****> const u,
        Kokkos::View<double***> const P,
        Kokkos::View<const double*> const nodes_x0,
        Kokkos::View<const double*> const nodes_y0,
        [[maybe_unused]]Kokkos::View<double*> g_array) const final
    {
        assert(rho.extent(0) == u.extent(0));
        assert(u.extent(0) == P.extent(0));
        assert(rho.extent(1) == u.extent(1));
        assert(u.extent(1) == P.extent(1));
        assert(rho.extent(2) == u.extent(2));
        assert(u.extent(2) == P.extent(2));

        double P0 = 5;

        auto const [begin, end] = cell_range(range);
        Kokkos::parallel_for(
        "GreshoVortexInit",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
        KOKKOS_CLASS_LAMBDA(int i, int j, int k)
        {
            rho(i, j, k) = 1;
            double x = nodes_x0(i);
            double y = nodes_y0(j);
            double r = std::sqrt(x * x + y * y);
            double theta = std::atan2(y, x);
            double u_theta;
            if (r < 0.2)
            {
                P(i, j, k) = P0 + 12.5 * r * r;
                u_theta = 5 * r;
                u(i, j, k, 0) = - u_theta * std::sin(theta);
                u(i, j, k, 1) = u_theta * std::cos(theta);
            }
            
            else if ((r >= 0.2) & (r < 0.4))
            {
                P(i, j, k) = P0 + 12.5 * r * r + 4 - 20 * r + 4 * std::log(5 * r);
                u_theta = 2 - 5 * r;
                u(i, j, k, 0) = - u_theta * std::sin(theta);
                u(i, j, k, 1) = u_theta * std::cos(theta);
            }
            else
            {
                P(i, j, k) = P0 - 2 + 4 * std::log(2);
                for (int idim = 0; idim < ndim; ++idim)
                {
                    u(i, j, k, idim) = 0;
                }
            } 
        });
    }
};

class RayleighTaylor : public IInitialisationProblem
{
public:
    void execute(
        Range const& range,
        Kokkos::View<double***> const rho,
        Kokkos::View<double****> const u,
        Kokkos::View<double***> const P,
        Kokkos::View<const double*> const nodes_x0,
        Kokkos::View<const double*> const nodes_y0,
        Kokkos::View<double*> g_array) const final
    {
        assert(rho.extent(0) == u.extent(0));
        assert(u.extent(0) == P.extent(0));
        assert(rho.extent(1) == u.extent(1));
        assert(u.extent(1) == P.extent(1));
        assert(rho.extent(2) == u.extent(2));
        assert(u.extent(2) == P.extent(2));

        int ng = 4;
        double Lx = nodes_x0(rho.extent(0)-ng) - nodes_x0(0);
        double Ly = nodes_y0(rho.extent(1)-ng) - nodes_y0(0);; // à revoir
        double g = g_array(1);
        double P0 = 2.5;
        double A = 0.01;

        auto const [begin, end] = cell_range(range);
        Kokkos::parallel_for(
        "RayleighTaylorInit",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
        KOKKOS_CLASS_LAMBDA(int i, int j, int k)
        {
            double x = nodes_x0(i);
            double y = nodes_y0(j);
            if (y >= 0)
            {
                rho(i, j, k) = 2;
                P(i, j, k) = P0 + y * g * rho(i, j, k);
            }
            if (y < 0)
            {
                rho(i, j, k) = 1;
                P(i, j, k) = P0 - y * g * rho(i, j, k);
            }
            u(i, j, k, 0) = 0;
            u(i, j, k, 1) = (A/4) * (1+std::cos(2*M_PI*x/Lx)) * (1+std::cos(2*M_PI*y/Ly));
         });
    }
}; 

inline std::unique_ptr<IInitialisationProblem> factory_initialisation(
    std::string const& problem)
{
    if (problem == "ShockTube")
    {
        return std::make_unique<ShockTube>();
    }
    if (problem == "AdvectionSinus")
    {
        return std::make_unique<AdvectionSinus>();
    }
    if (problem == "AdvectionCrenel")
    {
        return std::make_unique<AdvectionCrenel>();
    }
    if (problem == "GreshoVortex")
    {
        return std::make_unique<GreshoVortex>();
    }
    if (problem == "RayleighTaylor")
    {
        return std::make_unique<RayleighTaylor>();
    }
    throw std::runtime_error("Unknown problem for initialisation : " + problem + ".");
}

} // namespace novapp
