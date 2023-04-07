//!
//! @file initialisation_problem.hpp
//!

#pragma once

#include <Kokkos_Core.hpp>
#include <PerfectGas.hpp>

#include "euler_equations.hpp"
#include "ndim.hpp"
#include "range.hpp"
#include "Kokkos_shortcut.hpp"
#include "grid.hpp"

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
        KV_double_3d rho,
        KV_double_4d u,
        KV_double_3d P,
        [[maybe_unused]] KV_double_1d g,
        [[maybe_unused]] thermodynamics::PerfectGas const& eos,
        [[maybe_unused]] Grid const& grid) const 
        = 0;
};

class ShockTube : public IInitialisationProblem
{
public:
    void execute(
        Range const& range,
        KV_double_3d const rho,
        KV_double_4d const u,
        KV_double_3d const P,
        [[maybe_unused]] KV_double_1d g,
        [[maybe_unused]] thermodynamics::PerfectGas const& eos,
        [[maybe_unused]] Grid const& grid) const final
    {
        assert(rho.extent(0) == u.extent(0));
        assert(u.extent(0) == P.extent(0));
        assert(rho.extent(1) == u.extent(1));
        assert(u.extent(1) == P.extent(1));
        assert(rho.extent(2) == u.extent(2));
        assert(u.extent(2) == P.extent(2));

        auto const x_d = grid.x.d_view;
        auto const [begin, end] = cell_range(range);
        Kokkos::parallel_for(
        "ShockTubeInit",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
        KOKKOS_CLASS_LAMBDA(int i, int j, int k)
        {
            if((x_d(i) + x_d(i+1)) / 2 <= 0.5)
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
        KV_double_3d rho,
        KV_double_4d u,
        KV_double_3d P,
        [[maybe_unused]] KV_double_1d g,
        [[maybe_unused]] thermodynamics::PerfectGas const& eos,
        [[maybe_unused]] Grid const& grid) const final
    {
        assert(rho.extent(0) == u.extent(0));
        assert(u.extent(0) == P.extent(0));
        assert(rho.extent(1) == u.extent(1));
        assert(u.extent(1) == P.extent(1));
        assert(rho.extent(2) == u.extent(2));
        assert(u.extent(2) == P.extent(2));

        auto const x_d = grid.x.d_view;
        auto const [begin, end] = cell_range(range);
        Kokkos::parallel_for(
        "AdvectionInitSinus",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
        KOKKOS_CLASS_LAMBDA(int i, int j, int k)
        {
            rho(i, j, k) = 1 * Kokkos::exp(-15*Kokkos::pow(1./2 - (x_d(i)+x_d(i+1))/2, 2));
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
        KV_double_3d rho,
        KV_double_4d u,
        KV_double_3d P,
        [[maybe_unused]] KV_double_1d g,
        [[maybe_unused]] thermodynamics::PerfectGas const& eos,
        [[maybe_unused]] Grid const& grid) const final
    {
        assert(rho.extent(0) == u.extent(0));
        assert(u.extent(0) == P.extent(0));
        assert(rho.extent(1) == u.extent(1));
        assert(u.extent(1) == P.extent(1));
        assert(rho.extent(2) == u.extent(2));
        assert(u.extent(2) == P.extent(2));

        auto const x_d = grid.x.d_view;
        auto const [begin, end] = cell_range(range);
        Kokkos::parallel_for(
        "AdvectionInitCrenel",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
        KOKKOS_CLASS_LAMBDA(int i, int j, int k)
        {
            if ((x_d(i) + x_d(i+1)) / 2 <= 0.3)
            {
                rho(i, j, k) = 1;
            }
            else if ((x_d(i) + x_d(i+1)) / 2 >= 0.7)
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
        KV_double_3d const rho,
        KV_double_4d const u,
        KV_double_3d const P,
        [[maybe_unused]] KV_double_1d g,
        [[maybe_unused]] thermodynamics::PerfectGas const& eos,
        [[maybe_unused]] Grid const& grid) const final
    {
        assert(rho.extent(0) == u.extent(0));
        assert(u.extent(0) == P.extent(0));
        assert(rho.extent(1) == u.extent(1));
        assert(u.extent(1) == P.extent(1));
        assert(rho.extent(2) == u.extent(2));
        assert(u.extent(2) == P.extent(2));

        double P0 = 5;

        auto const x_d = grid.x.d_view;
        auto const y_d = grid.y.d_view;
        auto const [begin, end] = cell_range(range);
        Kokkos::parallel_for(
        "GreshoVortexInit",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
        KOKKOS_CLASS_LAMBDA(int i, int j, int k)
        {
            double x = x_d(i);
            double y = y_d(j);
            double r = Kokkos::sqrt(x * x + y * y);
            double theta = Kokkos::atan2(y, x);
            double u_theta;
            rho(i, j, k) = 1;
            if (r < 0.2)
            {
                P(i, j, k) = P0 + 12.5 * r * r;
                u_theta = 5 * r;
                u(i, j, k, 0) = - u_theta * Kokkos::sin(theta);
                u(i, j, k, 1) = u_theta * Kokkos::cos(theta);
            }
            
            else if ((r >= 0.2) && (r < 0.4))
            {
                P(i, j, k) = P0 + 12.5 * r * r + 4 - 20 * r + 4 * Kokkos::log(5 * r);
                u_theta = 2 - 5 * r;
                u(i, j, k, 0) = - u_theta * Kokkos::sin(theta);
                u(i, j, k, 1) = u_theta * Kokkos::cos(theta);
            }
            else
            {
                P(i, j, k) = P0 - 2 + 4 * Kokkos::log(2);
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
        KV_double_3d const rho,
        KV_double_4d const u,
        KV_double_3d const P,
        KV_double_1d g,
        [[maybe_unused]] thermodynamics::PerfectGas const& eos,
        [[maybe_unused]] Grid const& grid) const final
    {
        assert(rho.extent(0) == u.extent(0));
        assert(u.extent(0) == P.extent(0));
        assert(rho.extent(1) == u.extent(1));
        assert(u.extent(1) == P.extent(1));
        assert(rho.extent(2) == u.extent(2));
        assert(u.extent(2) == P.extent(2));

        double gval = g(1);
        double P0 = 2.5;
        double A = 0.01;

        auto const x_d = grid.x.d_view;
        auto const y_d = grid.y.d_view;
        auto const [begin, end] = cell_range(range);
        Kokkos::parallel_for(
        "RayleighTaylorInit",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
        KOKKOS_CLASS_LAMBDA(int i, int j, int k)
        {
            double x = x_d(i);
            double y = y_d(j);
            if (y >= 0)
            {
                rho(i, j, k) = 2;
            }
            if (y < 0)
            {
                rho(i, j, k) = 1;
            }
            P(i, j, k) = P0 + rho(i, j, k) * gval * y;
            u(i, j, k, 0) = 0;
            u(i, j, k, 1) = (A/4) * (1+Kokkos::cos(2*Kokkos::numbers::pi*x/grid.L[0])) * (1+Kokkos::cos(2*Kokkos::numbers::pi*y/grid.L[1]));
         });
    }
};

class SedovBlastWave2D : public IInitialisationProblem
{
public:
    void execute(
        Range const& range,
        KV_double_3d const rho,
        KV_double_4d const u,
        KV_double_3d const P,
        [[maybe_unused]] KV_double_1d g,
        thermodynamics::PerfectGas const& eos,
        [[maybe_unused]] Grid const& grid) const final
    {
        assert(rho.extent(0) == u.extent(0));
        assert(u.extent(0) == P.extent(0));
        assert(rho.extent(1) == u.extent(1));
        assert(u.extent(1) == P.extent(1));
        assert(rho.extent(2) == u.extent(2));
        assert(u.extent(2) == P.extent(2));

        double gamma = eos.compute_adiabatic_index();
        double E0 = 1E-12;
        double Eperturb = 1E5;

        auto const x_d = grid.x.d_view;
        auto const y_d = grid.y.d_view;
        auto const [begin, end] = cell_range(range);
        Kokkos::parallel_for(
        "SedovBlastWaveInit",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
        KOKKOS_CLASS_LAMBDA(int i, int j, int k)
        {
            double dv = grid.dx[0] * grid.dx[1];
            double x = x_d(i);
            double y = y_d(j);
            double r = Kokkos::sqrt(x * x + y * y);

            rho(i, j, k) = 1;
            for (int idim = 0; idim < ndim; ++idim)
            {
                u(i, j, k, idim) = 1;
            }
            if (r <0.025)
            {
                P(i, j, k) = Eperturb * (gamma - 1) / dv;
            }
            else
            {
                P(i, j, k) = E0 * (gamma - 1) / dv;
            } 
        });
    }
};

class SedovBlastWave1D : public IInitialisationProblem
{
public:
    void execute(
        Range const& range,
        KV_double_3d const rho,
        KV_double_4d const u,
        KV_double_3d const P,
        [[maybe_unused]] KV_double_1d g,
        thermodynamics::PerfectGas const& eos,
        [[maybe_unused]] Grid const& grid) const final
    {
        assert(rho.extent(0) == u.extent(0));
        assert(u.extent(0) == P.extent(0));
        assert(rho.extent(1) == u.extent(1));
        assert(u.extent(1) == P.extent(1));
        assert(rho.extent(2) == u.extent(2));
        assert(u.extent(2) == P.extent(2));

        double gamma = eos.compute_adiabatic_index();
        double E0 = 1E-12;
        double Eperturb = 0.5;
        double dv = grid.dx[0];

        auto const [begin, end] = cell_range(range);
        Kokkos::parallel_for(
        "SedovBlastWaveInit",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
        KOKKOS_CLASS_LAMBDA(int i, int j, int k)
        {
            rho(i, j, k) = 1;
            for (int idim = 0; idim < ndim; ++idim)
            {
                u(i, j, k, idim) = 0;
            }
            P(i, j, k) = E0 * (gamma - 1) / dv;
        });
        P(2, 0, 0) = Eperturb * (gamma - 1) /  dv;
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
    if (problem == "Sedov2d")
    {
        return std::make_unique<SedovBlastWave2D>();
    } 
    if (problem == "Sedov1d")
    {
        return std::make_unique<SedovBlastWave1D>();
    }    
    throw std::runtime_error("Unknown problem for initialisation : " + problem + ".");
}

} // namespace novapp
