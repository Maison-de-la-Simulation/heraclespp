// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cassert>
#include <iostream>
#include <string>
#include <utility>

#include <inih/INIReader.hpp>

#include <Kokkos_Core.hpp>
#include <grid.hpp>
#include <range.hpp>
#include <units.hpp>

#include "boundary.hpp"
#include "default_grid_setup.hpp" // IWYU pragma: keep
#include "default_shift_criterion.hpp" // IWYU pragma: keep
#include "default_user_step.hpp" // IWYU pragma: keep
#include "eos.hpp"
#include "initialization_interface.hpp"
#include "kokkos_shortcut.hpp"
#include "ndim.hpp"

namespace novapp
{

class ParamSetup
{
public:
    double rho0;
    double u0;
    double T;
    double gx;

    explicit ParamSetup(INIReader const& reader)
        : rho0(reader.GetReal("Initialisation", "rho0", 1.0))
        , u0(reader.GetReal("Initialisation", "u0", 1.0))
        , T(reader.GetReal("Perfect Gas", "temperature", 1.0))
        , gx(reader.GetReal("Gravity", "gx", 1.0))
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
        ParamSetup const& param_setup,
        Gravity gravity)
        : m_eos(eos)
        , m_param_setup(param_setup)
        , m_gravity(std::move(gravity))
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

        auto const xc = grid.x_center;
        auto const& eos = m_eos;
        auto const& gravity = m_gravity;
        auto const& param_setup = m_param_setup;
        double const mu = m_eos.mean_molecular_weight();
        std::cout <<"Scale = " << units::kb * m_param_setup.T
            / (mu * units::mp * m_param_setup.gx) << std::endl;

        Kokkos::parallel_for(
            "stratified_atm_init",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                double const x0 = units::kb * param_setup.T
                        / (mu * units::mp * Kokkos::fabs(gravity(i, j, k, 0)));

                rho(i, j, k) = param_setup.rho0 * Kokkos::exp(- xc(i) / x0);

                for (int idim = 0; idim < ndim; ++idim)
                {
                    u(i, j, k, idim) = param_setup.u0;
                }

                P(i, j, k) = eos.compute_P_from_T(rho(i, j, k), param_setup.T);
            });
    }
};

template <class Gravity>
class BoundarySetup : public IBoundaryCondition<Gravity>
{
    std::string m_label;

private:
    EOS m_eos;
    ParamSetup m_param_setup;

public:
    BoundarySetup(int idim, int iface,
        [[maybe_unused]] EOS const& eos,
        [[maybe_unused]] ParamSetup const& param_setup)
        : IBoundaryCondition<Gravity>(idim, iface)
        , m_label(std::string("UserDefined").append(bc_dir(idim)).append(bc_face(iface)))
        , m_eos(eos)
        , m_param_setup(param_setup)
    {
    }

    void execute(Grid const& grid,
                 [[maybe_unused]] Gravity const& gravity,
                 KV_double_3d const& rho,
                 KV_double_4d const& rhou,
                 KV_double_3d const& E,
                 [[maybe_unused]] KV_double_4d const& fx) const final
    {
        assert(rho.extent(0) == rhou.extent(0));
        assert(rhou.extent(0) == E.extent(0));
        assert(rho.extent(1) == rhou.extent(1));
        assert(rhou.extent(1) == E.extent(1));
        assert(rho.extent(2) == rhou.extent(2));
        assert(rhou.extent(2) == E.extent(2));

        Kokkos::Array<int, 3> begin {0, 0, 0};
        Kokkos::Array<int, 3> end {rho.extent_int(0), rho.extent_int(1), rho.extent_int(2)};

        auto const xc = grid.x_center;
        auto const& eos = m_eos;
        auto const& param_setup = m_param_setup;
        double const mu = m_eos.mean_molecular_weight();

        int const ng = grid.Nghost[this->bc_idim()];
        if (this->bc_iface() == 1)
        {
            begin[this->bc_idim()] = rho.extent_int(this->bc_idim()) - ng;
        }
        end[this->bc_idim()] = begin[this->bc_idim()] + ng;

        Kokkos::parallel_for(
            m_label,
            Kokkos::MDRangePolicy<Kokkos::IndexType<int>, Kokkos::Rank<3>>(begin, end),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                double const gravity_x = gravity(i, j, k, 0);

                double const x0 = units::kb * param_setup.T * units::Kelvin
                        / (mu * units::mp * Kokkos::fabs(gravity_x));

                rho(i, j, k) = param_setup.rho0 * Kokkos::exp(- xc(i) / x0);

                for (int n = 0; n < rhou.extent_int(3); ++n)
                {
                    rhou(i, j, k, n) = param_setup.rho0 * param_setup.u0;
                }

                E(i, j, k) = eos.compute_evol_from_T(rho(i, j, k), param_setup.T);
            });
    }
};

} // namespace novapp
