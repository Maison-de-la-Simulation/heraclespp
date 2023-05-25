
#pragma once

#include <Kokkos_Core.hpp>
#include <PerfectGas.hpp>

#include <inih/INIReader.hpp>

#include "euler_equations.hpp"
#include "ndim.hpp"
#include "range.hpp"
#include "Kokkos_shortcut.hpp"
#include "grid.hpp"
#include "units.hpp"
#include "initialization_interface.hpp"

namespace novapp
{

class ParamSetup
{
public:
    double rho0;
    double rho1;
    double u0;
    double P0;

    explicit ParamSetup(INIReader const& reader)
        : rho0(reader.GetReal("Initialisation", "rho0", 1.0))
        , rho1(reader.GetReal("Initialisation", "rho1", 1.0))
        , u0(reader.GetReal("Initialisation", "u0", 1.0))
        , P0(reader.GetReal("Initialisation", "P0", 1.0))
    {
    }
};

class InitializationSetup : public IInitializationProblem
{
private:
    thermodynamics::PerfectGas m_eos;
    Grid m_grid;
    ParamSetup m_param_setup;

public:
    InitializationSetup(
        thermodynamics::PerfectGas const& eos,
        Grid const& grid,
        ParamSetup const& param_set_up)
        : m_eos(eos)
        , m_grid(grid)
        , m_param_setup(param_set_up)
    {
    }

    void execute(
        Range const& range,
        KV_double_3d const rho,
        KV_double_4d const u,
        KV_double_3d const P,
        [[maybe_unused]] KV_double_4d const fx,
        [[maybe_unused]] KV_double_1d g) const final
    {
        assert(rho.extent(0) == u.extent(0));
        assert(u.extent(0) == P.extent(0));
        assert(rho.extent(1) == u.extent(1));
        assert(u.extent(1) == P.extent(1));
        assert(rho.extent(2) == u.extent(2));
        assert(u.extent(2) == P.extent(2));

        auto const x_d = m_grid.x.d_view;
        auto const [begin, end] = cell_range(range);
        Kokkos::parallel_for(
        "advection_gap_init",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
        KOKKOS_CLASS_LAMBDA(int i, int j, int k)
        {
            if ((x_d(i) + x_d(i+1)) * units::m / 2 <= 0.3)
            {
                rho(i, j, k) = m_param_setup.rho0 * units::density;
            }
            else if ((x_d(i) + x_d(i+1)) * units::m / 2 >= 0.7)
            {
                rho(i, j, k) = m_param_setup.rho0 * units::density;
            }
            else
            {
                rho(i, j, k) = m_param_setup.rho1 * units::density;
            }
            P(i, j, k) = m_param_setup.P0 * units::pressure;
            for (int idim = 0; idim < ndim; ++idim)
            {
                u(i, j, k, idim) = m_param_setup.u0 * units::velocity;
            }
        });  
    }
};

class BoundarySetup : public IBoundaryCondition
{
public:
    BoundarySetup(int idim, int iface,
        [[maybe_unused]] thermodynamics::PerfectGas const& eos,
        [[maybe_unused]] Grid const& grid,
        [[maybe_unused]] ParamSetup const& param_setup)
        : IBoundaryCondition(idim, iface)
    {
    }
};

} // namespace novapp
