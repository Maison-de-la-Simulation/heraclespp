// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <string>

#include <inih/INIReader.hpp>

#include <Kokkos_Core.hpp>
#include <PerfectGas.hpp>
#include <grid.hpp>
#include <kokkos_shortcut.hpp>
#include <ndim.hpp>
#include <range.hpp>

#include "default_boundary_setup.hpp" // IWYU pragma: keep
#include "default_grid_setup.hpp" // IWYU pragma: keep
#include "default_shift_criterion.hpp" // IWYU pragma: keep
#include "default_user_step.hpp" // IWYU pragma: keep
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
    ParamSetup m_param_setup;

public:
    InitializationSetup(
        thermodynamics::PerfectGas const& eos,
        ParamSetup const& param_set_up,
        [[maybe_unused]] Gravity const& gravity)
        : m_eos(eos)
        , m_param_setup(param_set_up)
    {
    }

    void execute(
        Range const& range,
        Grid const& grid,
        KV_double_3d const& rho,
        KV_double_4d const& u,
        KV_double_3d const& P,
        [[maybe_unused]] KV_double_4d const& fx) const final
    {
        assert(equal_extents({0, 1, 2}, rho, u, P, fx));
        assert(u.extent_int(3) == ndim);
        assert(fx.extent_int(3) == 0);

        auto const xc = grid.x_center;
        auto const yc = grid.y_center;
        auto const& param_setup = m_param_setup;

        Kokkos::parallel_for(
            "implosion_test_init",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                double const x = xc(i);
                double const y = yc(j);

                if (x + y >  0.15)
                {
                    rho(i, j, k) = param_setup.rho0;
                    P(i, j, k) = param_setup.P0;
                }
                else
                {
                    rho(i, j, k) = param_setup.rho1;
                    P(i, j, k) = param_setup.P1;
                }

                for (int idim = 0; idim < ndim; ++idim)
                {
                    u(i, j, k, idim) = param_setup.u0;
                }
            });
    }
};

} // namespace novapp
