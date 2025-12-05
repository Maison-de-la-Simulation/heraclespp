// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <string>

#include <inih/INIReader.hpp>

#include <Kokkos_Core.hpp>
#include <eos.hpp>
#include <grid.hpp>
#include <kokkos_shortcut.hpp>
#include <ndim.hpp>
#include <range.hpp>

#include "default_boundary_setup.hpp" // IWYU pragma: keep
#include "default_grid_setup.hpp" // IWYU pragma: keep
#include "default_shift_criterion.hpp" // IWYU pragma: keep
#include "default_user_step.hpp" // IWYU pragma: keep
#include "initialization_interface.hpp"

namespace hclpp {

class ParamSetup
{
public:
    double u0;
    double P0;

    explicit ParamSetup(INIReader const& reader) : u0(reader.GetReal("Initialisation", "u0", 1.0)), P0(reader.GetReal("Initialisation", "P0", 1.0)) {}
};

template <class Gravity>
class InitializationSetup : public IInitializationProblem
{
private:
    ParamSetup m_param_setup;

public:
    InitializationSetup(EOS const& /*eos*/, ParamSetup const& param_set_up, Gravity const& /*gravity*/) : m_param_setup(param_set_up) {}

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

        auto const xc = grid.x0_center;
        auto const& param_setup = m_param_setup;

        double const rho0 = 1.;
        double const drho0 = 1E-1;

        Kokkos::parallel_for(
                "advection_sinus_init",
                cell_mdrange(range),
                KOKKOS_LAMBDA(int i, int j, int k) {
                    rho(i, j, k) = rho0 + (drho0 * Kokkos::sin(2 * Kokkos::numbers::pi * xc(i)));

                    for (int idim = 0; idim < ndim; ++idim) {
                        u(i, j, k, idim) = param_setup.u0;
                    }

                    P(i, j, k) = param_setup.P0;
                });
    }
};

} // namespace hclpp
