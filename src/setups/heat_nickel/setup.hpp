// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <string>

#include <inih/INIReader.hpp>

#include <Kokkos_Core.hpp>
#include <eos.hpp>
#include <kokkos_shortcut.hpp>
#include <ndim.hpp>
#include <range.hpp>
#include <units.hpp>

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
    double u0;
    double T0;
    double fx0;

    explicit ParamSetup(INIReader const& reader)
        : rho0(reader.GetReal("Initialisation", "rho0", 1.0))
        , u0(reader.GetReal("Initialisation", "u0", 1.0))
        , T0(reader.GetReal("Initialisation", "T0", 1.0))
        , fx0(reader.GetReal("Initialisation", "fx0", 1.0))
    {
    }
};

template <class Gravity>
class InitializationSetup : public IInitializationProblem
{
private:
    EOS m_eos;
    ParamSetup m_param_setup;

public:
    InitializationSetup(
        EOS const& eos,
        ParamSetup const& param_set_up,
        Gravity const& /*gravity*/)
        : m_eos(eos)
        , m_param_setup(param_set_up)
    {
    }

    void execute(
        Range const& range,
        Grid const& /*grid*/,
        KV_double_3d const& rho,
        KV_double_4d const& u,
        KV_double_3d const& P,
        KV_double_4d const& fx) const final
    {
        assert(equal_extents({0, 1, 2}, rho, u, P, fx));
        assert(u.extent_int(3) == ndim);
        assert(fx.extent_int(3) == 1);

        auto const& eos = m_eos;
        auto const& param_setup = m_param_setup;

        Kokkos::parallel_for(
            "heat_nickel_init",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                rho(i, j, k) = param_setup.rho0 * units::density;

                P(i, j, k) = eos.compute_pres_from_temp(rho(i, j, k), param_setup.T0) * units::pressure;

                for (int idim = 0; idim < ndim; ++idim)
                {
                    u(i, j, k, idim) = param_setup.u0 * units::velocity;
                }

                fx(i, j, k, 0) = param_setup.fx0;
            });
    }
};

} // namespace novapp
