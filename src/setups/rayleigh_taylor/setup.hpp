// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

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
    double u;
    double A;
    double fx0;
    double fx1;

    explicit ParamSetup(INIReader const& reader)
        : rho0(reader.GetReal("Initialisation", "rho0", 1.0))
        , rho1(reader.GetReal("Initialisation", "rho1", 1.0))
        , u(reader.GetReal("Initialisation", "u", 1.0))
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
        Gravity gravity)
        : m_eos(eos)
        , m_param_setup(param_set_up)
        , m_gravity(std::move(gravity))
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

        double const P0 = (10. / 7 + 1. / 4);

        auto const xc = grid.x_center;
        auto const yc = grid.y_center;
        auto const& gravity = m_gravity;
        auto const& param_setup = m_param_setup;

        Kokkos::parallel_for(
            "Rayleigh_Taylor_2D_init",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                double const x = xc(i);
                double const y = yc(j);
                double const h = 0.01 * Kokkos::cos(4 * Kokkos::numbers::pi * x);

                rho(i, j, k) = param_setup.rho1 + (param_setup.rho0 - param_setup.rho1) / 2
                                * (1 + Kokkos::tanh((y - h) / 0.005));

                fx(i, j, k, 0) = (param_setup.fx0 - param_setup.fx1) / 2
                                * (1 + Kokkos::tanh((y - h) / 0.005));

                u(i, j, k, 0) = param_setup.u;
                u(i, j, k, 1) = param_setup.u;

                P(i, j, k) = P0 - param_setup.rho0 * Kokkos::fabs(gravity(i, j, k, 1)) * y;
            });
    }
};

} // namespace novapp
