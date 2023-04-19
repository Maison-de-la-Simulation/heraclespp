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
#include "units.hpp"

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
        KV_double_1d g,
        thermodynamics::PerfectGas const& eos,
        Grid const& grid,
        Param const& param) const 
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
        [[maybe_unused]] Grid const& grid,
        [[maybe_unused]] Param const& param) const final
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
                rho(i, j, k) = param.rho0;
                P(i, j, k) = param.P0;
                for (int idim = 0; idim < ndim; ++idim)
                {
                    u(i, j, k, idim) = param.u0;
                }
            }
            else
            {
                rho(i, j, k) = param.rho1;
                P(i, j, k) =  param.P1;
                for (int idim = 0; idim < ndim; ++idim)
                {
                    u(i, j, k, idim) = param.u1;
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
        [[maybe_unused]] Grid const& grid,
        Param const& param) const final
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
            rho(i, j, k) = param.rho0 * Kokkos::exp(-15*Kokkos::pow(1./2 - (x_d(i)+x_d(i+1))/2, 2));
            P(i, j, k) = param.P0;
            for (int idim = 0; idim < ndim; ++idim)
            {
                u(i, j, k, idim) = param.u0;
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
        [[maybe_unused]] Grid const& grid,
        Param const& param) const final
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
                rho(i, j, k) = param.rho0;
            }
            else if ((x_d(i) + x_d(i+1)) / 2 >= 0.7)
            {
                rho(i, j, k) = param.rho0;
            }
            else
            {
                rho(i, j, k) = param.rho1;
            }
            P(i, j, k) = param.P0;
            for (int idim = 0; idim < ndim; ++idim)
            {
                u(i, j, k, idim) = param.u0;
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
        [[maybe_unused]] Grid const& grid,
        Param const& param) const final
    {
        assert(rho.extent(0) == u.extent(0));
        assert(u.extent(0) == P.extent(0));
        assert(rho.extent(1) == u.extent(1));
        assert(u.extent(1) == P.extent(1));
        assert(rho.extent(2) == u.extent(2));
        assert(u.extent(2) == P.extent(2));

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
            rho(i, j, k) = param.rho0;
            if (r < 0.2)
            {
                P(i, j, k) = param.P0 + 12.5 * r * r;
                u_theta = 5 * r;
                u(i, j, k, 0) = - u_theta * Kokkos::sin(theta);
                u(i, j, k, 1) = u_theta * Kokkos::cos(theta);
            }
            
            else if ((r >= 0.2) && (r < 0.4))
            {
                P(i, j, k) = param.P0 + 12.5 * r * r + 4 - 20 * r + 4 * Kokkos::log(5 * r);
                u_theta = 2 - 5 * r;
                u(i, j, k, 0) = - u_theta * Kokkos::sin(theta);
                u(i, j, k, 1) = u_theta * Kokkos::cos(theta);
            }
            else
            {
                P(i, j, k) = param.P0 - 2 + 4 * Kokkos::log(2);
                for (int idim = 0; idim < ndim; ++idim)
                {
                    u(i, j, k, idim) = param.u0;
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
        Grid const& grid,
        Param const& param) const final
    {
        assert(rho.extent(0) == u.extent(0));
        assert(u.extent(0) == P.extent(0));
        assert(rho.extent(1) == u.extent(1));
        assert(u.extent(1) == P.extent(1));
        assert(rho.extent(2) == u.extent(2));
        assert(u.extent(2) == P.extent(2));

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
                rho(i, j, k) = param.rho1;
            }
            if (y < 0)
            {
                rho(i, j, k) = param.rho0;
            }
            u(i, j, k, 0) = param.u0;
            u(i, j, k, 1) = (param.A/4) * (1+Kokkos::cos(2*Kokkos::numbers::pi*x/grid.L[0])) * (1+Kokkos::cos(2*Kokkos::numbers::pi*y/grid.L[1]));
            P(i, j, k) = param.P0 + rho(i, j, k) * g(1) * y;
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
        Grid const& grid,
        Param const& param) const final
    {
        assert(rho.extent(0) == u.extent(0));
        assert(u.extent(0) == P.extent(0));
        assert(rho.extent(1) == u.extent(1));
        assert(u.extent(1) == P.extent(1));
        assert(rho.extent(2) == u.extent(2));
        assert(u.extent(2) == P.extent(2));

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

            rho(i, j, k) = param.rho0;
            for (int idim = 0; idim < ndim; ++idim)
            {
                u(i, j, k, idim) = param.u0;
            }
            if (r <0.025)
            {
                P(i, j, k) = eos.compute_pressure_from_e(rho(i, j, k), param.E1 / dv);
            }
            else
            {
                P(i, j, k) = eos.compute_pressure_from_e(rho(i, j, k), param.E0 / dv);
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
        [[maybe_unused]] Grid const& grid,
        Param const& param) const final
    {
        assert(rho.extent(0) == u.extent(0));
        assert(u.extent(0) == P.extent(0));
        assert(rho.extent(1) == u.extent(1));
        assert(u.extent(1) == P.extent(1));
        assert(rho.extent(2) == u.extent(2));
        assert(u.extent(2) == P.extent(2));

        double dv = grid.dx[0];

        auto const [begin, end] = cell_range(range);
        Kokkos::parallel_for(
        "SedovBlastWaveInit",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
        KOKKOS_CLASS_LAMBDA(int i, int j, int k)
        {
            rho(i, j, k) = param.rho0;
            for (int idim = 0; idim < ndim; ++idim)
            {
                u(i, j, k, idim) = param.u0;
            }
            P(i, j, k) = eos.compute_pressure_from_e(rho(i, j, k), param.E0 / dv);
            if(grid.mpi_rank==0)
            {
                P(2, j, k) = eos.compute_pressure_from_e(rho(i, j, k), param.E1 / dv);
            }
        });
    }
};

class StratifiedAtm1D : public IInitialisationProblem
{
public:
    void execute(
        Range const& range,
        KV_double_3d const rho,
        KV_double_4d const u,
        KV_double_3d const P,
        KV_double_1d g,
        thermodynamics::PerfectGas const& eos,
        Grid const& grid,
        Param const& param) const final
    {
        assert(rho.extent(0) == u.extent(0));
        assert(u.extent(0) == P.extent(0));
        assert(rho.extent(1) == u.extent(1));
        assert(u.extent(1) == P.extent(1));
        assert(rho.extent(2) == u.extent(2));
        assert(u.extent(2) == P.extent(2));

        auto const x_d = grid.x.d_view;
        double mu = eos.compute_mean_molecular_weight();
        //std::cout <<"Scale = " << kb * T / (mu * mh * std::abs(gx)) << std::endl;
        
        auto const [begin, end] = cell_range(range);
        Kokkos::parallel_for(
        "StratifiedAtmInit",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
        KOKKOS_CLASS_LAMBDA(int i, int j, int k)
        {
            double xcenter = x_d(i) + grid.dx[0] / 2;
            double x0 = units::kb * param.T / (mu * units::mh * std::abs(g(0)));
            rho(i, j, k) = param.rho0 * Kokkos::exp(- xcenter / x0);
            for (int idim = 0; idim < ndim; ++idim)
            {
                u(i, j, k, idim) = param.u0;
            }
            P(i, j, k) = eos.compute_pressure_from_T(rho(i, j, k), param.T);
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
    if (problem == "Sedov2d")
    {
        return std::make_unique<SedovBlastWave2D>();
    }
    if (problem == "Sedov1d")
    {
        return std::make_unique<SedovBlastWave1D>();
    }
    if (problem == "StratAtm")
    {
        return std::make_unique<StratifiedAtm1D>();
    }
    throw std::runtime_error("Unknown problem for initialisation : " + problem + ".");
}

} // namespace novapp
