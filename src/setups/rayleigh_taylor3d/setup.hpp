// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <mpi.h>

#include <array>
#include <cassert>
#include <numeric>
#include <random>
#include <string>
#include <utility>

#include <inih/INIReader.hpp>

#include <Kokkos_Core.hpp>
#include <PerfectGas.hpp>
#include <grid.hpp>
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

template <class Gravity>
class InitializationSetup : public IInitializationProblem
{
private:
    thermodynamics::PerfectGas m_eos;
    ParamSetup m_param_setup;
    Gravity m_gravity;

public:
    InitializationSetup(
        thermodynamics::PerfectGas const& eos,
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
        assert(fx.extent_int(3) == 1);

        std::array<double, 4> data_to_broadcast;

        int const bcast_root = 0;

        if (grid.mpi_rank == bcast_root)
        {
            std::random_device rd;
            std::mt19937 const gen(rd());
            std::uniform_real_distribution<double> dist(-1.0, 1.0);

            for(double& data : data_to_broadcast)
            {
                data = dist(rd);
            }
        }

        MPI_Bcast(data_to_broadcast.data(), 4, MPI_DOUBLE, bcast_root, grid.comm_cart);

        double const ak = data_to_broadcast[0];
        double const bk = data_to_broadcast[1];
        double const ck = data_to_broadcast[2];
        double const dk = data_to_broadcast[3];

        Kokkos::Array<int, 5> kx_array;
        std::iota(Kokkos::begin(kx_array), Kokkos::end(kx_array), 0);
        Kokkos::Array<int, 5> ky_array;
        std::iota(Kokkos::begin(ky_array), Kokkos::end(ky_array), 0);

        double const L = 10;
        double const gamma = 5. / 3;
        // double hrms = 3E-4 * L;
        // double H = Kokkos::sqrt((1. / 4) * (ak * ak + bk * bk + ck * ck + dk * dk)) / hrms;

        auto const x_d = grid.x;
        auto const y_d = grid.y;
        auto const z_d = grid.z;
        auto const& gravity = m_gravity;
        auto const& param_setup = m_param_setup;

        int const Kmin = 8;
        int const Kmax = 16;

        Kokkos::parallel_for(
            "Rayleigh_Taylor_3D_init",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                double const P0 = (2 * units::pi * (param_setup.rho0 + param_setup.rho1)
                    * Kokkos::fabs(gravity(i, j, k, 2)) * L) * units::pressure;

                double const x = x_d(i) * units::m;
                double const y = y_d(j) * units::m;
                double const z = z_d(k) * units::m;

                double const X = 2 * units::pi * x / L;
                double const Y = 2 * units::pi * y / L;
                double h = 0;

                for (int const kx : kx_array)
                {
                    for (int const ky : ky_array)
                    {
                        int const K = (kx * kx) + (ky * ky);
                        if (K >= Kmin && K <= Kmax)
                        {
                            h += ((ak * Kokkos::cos(kx * X) * Kokkos::cos(ky * Y))
                        + (bk * Kokkos::cos(kx * X) * Kokkos::sin(ky * Y))
                        + (ck * Kokkos::sin(kx * X) * Kokkos::cos(ky * Y))
                        + (dk * Kokkos::sin(kx * X) * Kokkos::sin(ky * Y)));
                        }
                    }
                }

                if(z >= h)
                {
                    rho(i, j, k) = param_setup.rho0 * Kokkos::pow(1 - ((gamma - 1) / gamma
                                * (param_setup.rho0 * Kokkos::fabs(gravity(i, j, k, 2)) * z) / P0), 1. / (gamma - 1)) * units::density;

                    P(i, j, k) = P0 * Kokkos::pow(rho(i, j, k) / param_setup.rho0, gamma) * units::pressure;

                    fx(i, j, k, 0) = 1;
                }
                else
                {
                    rho(i, j, k) = param_setup.rho1 * Kokkos::pow(1 - ((gamma - 1) / gamma
                                * (param_setup.rho1 * Kokkos::fabs(gravity(i, j, k, 2)) * z) / P0), 1. / (gamma - 1)) * units::density;

                    P(i, j, k) = P0 * Kokkos::pow(rho(i, j, k) / param_setup.rho1, gamma) * units::pressure;

                    fx(i, j, k, 0) = 0;
                }

                for (int idim = 0; idim < ndim; ++idim)
                {
                    u(i, j, k, idim) = param_setup.u0 * units::velocity;
                }
            });
    }
};

} // namespace novapp
