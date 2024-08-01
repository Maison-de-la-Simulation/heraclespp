
#pragma once

#include <Kokkos_Core.hpp>
#include <units.hpp>

#include <inih/INIReader.hpp>

#include "default_boundary_setup.hpp"
#include "default_grid_setup.hpp"
#include "default_shift_criterion.hpp"
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
    double A;
    double fx0;
    double fx1;

    explicit ParamSetup(INIReader const& reader)
        : rho0(reader.GetReal("Initialisation", "rho0", 1.0))
        , rho1(reader.GetReal("Initialisation", "rho1", 1.0))
        , u0(reader.GetReal("Initialisation", "u0", 1.0))
        , P0(reader.GetReal("Initialisation", "P0", 1.0))
        , A(reader.GetReal("Initialisation", "A", 1.0))
        , fx0(reader.GetReal("Initialisation", "fx0", 1.0))
        , fx1(reader.GetReal("Initialisation", "fx1", 1.0))
    {
    }
};

template <class Gravity>
class InitializationSetup : public IInitializationProblem
{
private:
    EOS m_eos;
    ParamSetup m_param_setup;
    Gravity m_gravity;

public:
    InitializationSetup(
        EOS const& eos,
        ParamSetup const& param_set_up,
        Gravity const& gravity)
        : m_eos(eos)
        , m_param_setup(param_set_up)
        , m_gravity(gravity)
    {
    }

    void execute(
        Range const& range,
        Grid const& grid,
        KV_double_3d const& rho,
        KV_double_4d const& u,
        KV_double_3d const& P,
        KV_double_4d const& fx) const final
    {
        assert(equal_extents({0, 1, 2}, rho, u, P, fx));
        assert(u.extent_int(3) == ndim);

        double P0 = (10. / 7 + 1. / 4) * units::pressure;

        auto const x_d = grid.x;
        auto const y_d = grid.y;
        auto const& gravity = m_gravity;
        auto const& param_setup = m_param_setup;

        Kokkos::parallel_for(
            "Rayleigh_Taylor_2D_init",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                double x = x_d(i) * units::m;
                double y = y_d(j) * units::m;
                double h = 0.01 * Kokkos::cos(4 * Kokkos::numbers::pi * x);

                if (y >= h)
                {
                    rho(i, j, k) = param_setup.rho1 * units::density;
                    fx(i, j, k, 0) = param_setup.fx0;
                }

                if (y < h)
                {
                    rho(i, j, k) = param_setup.rho0 * units::density;
                    fx(i, j, k, 0) = param_setup.fx1;
                }

                u(i, j, k, 0) = param_setup.u0 * units::velocity;
                u(i, j, k, 1) = param_setup.u0 * units::velocity;
                /* u(i, j, k, 1) = (param_setup.A/4) * (1+Kokkos::cos(2*Kokkos::numbers::pi*x/grid.L[0]))
                                * (1+Kokkos::cos(2*Kokkos::numbers::pi*y/grid.L[1])); */

                P(i, j, k) = (P0 + rho(i, j, k) * gravity(i, j, k, 1) * units::acc * y) * units::pressure;
            });
    }
};

} // namespace novapp
