// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <string>

#include <inih/INIReader.hpp>

#include <Kokkos_Core.hpp>
#include <eos.hpp>
#include <geom.hpp>
#include <grid.hpp>
#include <kokkos_shortcut.hpp>
#include <ndim.hpp>
#include <range.hpp>

#include "default_boundary_setup.hpp" // IWYU pragma: keep
#include "default_grid_setup.hpp" // IWYU pragma: keep
#include "default_shift_criterion.hpp" // IWYU pragma: keep
#include "default_user_step.hpp" // IWYU pragma: keep
#include "initialization_interface.hpp"

namespace novapp {

class ParamSetup
{
public:
    double rho0;
    double u0;
    double E0;
    double E1;

    explicit ParamSetup(INIReader const& reader)
        : rho0(reader.GetReal("Initialisation", "rho0", 1.0))
        , u0(reader.GetReal("Initialisation", "u0", 1.0))
        , E0(reader.GetReal("Initialisation", "E0", 1.0))
        , E1(reader.GetReal("Initialisation", "E1", 1.0))
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
    InitializationSetup(EOS const& eos, ParamSetup const& param_set_up, Gravity const& /*gravity*/) : m_eos(eos), m_param_setup(param_set_up) {}

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

        int const mpi_rank = grid.mpi_rank;

        double alpha = -1;

        if (geom == Geometry::Geom_cartesian) {
            alpha = 0.5;
        }

        if (geom == Geometry::Geom_spherical) {
            alpha = 1;
        }

        auto const& eos = m_eos;
        auto const& param_setup = m_param_setup;
        auto const dv = grid.dv;

        Kokkos::parallel_for(
                "Sedov_1D_init",
                cell_mdrange(range),
                KOKKOS_LAMBDA(int i, int j, int k) {
                    double const dv_beg = dv(2, 0, 0);
                    rho(i, j, k) = param_setup.rho0;

                    for (int idim = 0; idim < ndim; ++idim) {
                        u(i, j, k, idim) = param_setup.u0;
                    }

                    double const T = eos.compute_temp_from_evol(rho(i, j, k), param_setup.E0 / dv_beg);
                    P(i, j, k) = eos.compute_pres_from_temp(rho(i, j, k), T);

                    if (mpi_rank == 0 && i == 2) {
                        double const T_perturb = eos.compute_temp_from_evol(rho(i, j, k), alpha * param_setup.E1 / dv_beg);
                        P(i, j, k) = eos.compute_pres_from_temp(rho(i, j, k), T_perturb);
                    }
                });
    }
};

} // namespace novapp
