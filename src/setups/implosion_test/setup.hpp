
#pragma once

#include <Kokkos_Core.hpp>
#include <units.hpp>

#include <inih/INIReader.hpp>

#include "default_boundary_setup.hpp"
#include "default_grid_setup.hpp"
#include "default_user_step.hpp"
#include "eos.hpp"
#include <grid.hpp>
#include "initialization_interface.hpp"
#include "kokkos_shortcut.hpp"
#include "ndim.hpp"
#include "nova_params.hpp"
#include <range.hpp>

namespace novapp
{

class ParamSetup
{
public:
    double rho0;
    double rho1;
    double u0;
    double P0;
    double P1;

    explicit ParamSetup(INIReader const& reader)
        : rho0(reader.GetReal("Initialisation", "rho0", 1.0))
        , rho1(reader.GetReal("Initialisation", "rho1", 1.0))
        , u0(reader.GetReal("Initialisation", "u0", 1.0))
        , P0(reader.GetReal("Initialisation", "P0", 1.0))
        , P1(reader.GetReal("Initialisation", "P1", 1.0))
    {
    }
};

template <class Gravity>
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
        ParamSetup const& param_set_up,
        [[maybe_unused]] Gravity const& gravity)
        : m_eos(eos)
        , m_grid(grid)
        , m_param_setup(param_set_up)
    {
    }

    void execute(
        Range const& range,
        KV_double_3d const& rho,
        KV_double_4d const& u,
        KV_double_3d const& P,
        [[maybe_unused]] KV_double_4d const& fx) const final
    {
        assert(rho.extent(0) == u.extent(0));
        assert(u.extent(0) == P.extent(0));
        assert(rho.extent(1) == u.extent(1));
        assert(u.extent(1) == P.extent(1));
        assert(rho.extent(2) == u.extent(2));
        assert(u.extent(2) == P.extent(2));

        auto const x_d = m_grid.x;
        auto const y_d = m_grid.y;
        auto const& param_setup = m_param_setup;

        Kokkos::parallel_for(
            "implosion_test_init",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                double x = x_d(i) * units::m;
                double y = y_d(j) * units::m;

                if (x + y >  0.15)
                {
                    rho(i, j, k) = param_setup.rho0 * units::density;

                    P(i, j, k) = param_setup.P0 * units::pressure;
                }

                else
                {
                    rho(i, j, k) = param_setup.rho1 * units::density;

                    P(i, j, k) = param_setup.P1 * units::pressure;
                }

                for (int idim = 0; idim < ndim; ++idim)
                {
                    u(i, j, k, idim) = param_setup.u0 * units::velocity;
                }
            });
    }
};

} // namespace novapp
