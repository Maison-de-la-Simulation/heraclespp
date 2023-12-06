
#pragma once

#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <units.hpp>
#include <random>

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
    Grid m_grid;
    ParamSetup m_param_setup;
    Gravity m_gravity;

public:
    InitializationSetup(
        thermodynamics::PerfectGas const& eos,
        Grid const& grid,
        ParamSetup const& param_set_up,
        Gravity const& gravity)
        : m_eos(eos)
        , m_grid(grid)
        , m_param_setup(param_set_up)
        , m_gravity(gravity)
    {
    }

    void execute(
        Range const& range,
        KV_double_3d const rho,
        KV_double_4d const u,
        KV_double_3d const P,
        KV_double_4d const fx) const final
    {
        assert(rho.extent(0) == u.extent(0));
        assert(u.extent(0) == P.extent(0));
        assert(rho.extent(1) == u.extent(1));
        assert(u.extent(1) == P.extent(1));
        assert(rho.extent(2) == u.extent(2));
        assert(u.extent(2) == P.extent(2));

        int mpi_rank = m_grid.mpi_rank;
        MPI_Comm comm_cart = m_grid.comm_cart;

        std::array<double, 4> data_to_broadcast;

        if (mpi_rank == 0)
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<double> dist(-1.0, 1.0);

            for(int i = 0; i < data_to_broadcast.size() ; ++i)
            {
                data_to_broadcast[i] = dist(rd);
            }
        }

        MPI_Bcast(data_to_broadcast.data(), 4, MPI_DOUBLE, 0, comm_cart);

        double ak = data_to_broadcast[0];
        double bk = data_to_broadcast[1];
        double ck = data_to_broadcast[2];
        double dk = data_to_broadcast[3];

        /* std::cout << "[MPI process "<< mpi_rank <<"] ak = " << ak << std::endl;
        std::cout << "[MPI process "<< mpi_rank <<"] bk = " << bk << std::endl;
        std::cout << "[MPI process "<< mpi_rank <<"] ck = " << ck << std::endl;
        std::cout << "[MPI process "<< mpi_rank <<"] dk = " << dk << std::endl; */

        Kokkos::Array<int, 5> kx;
        Kokkos::Array<int, 5> ky;
        for (int i = 0; i < kx.size(); ++i)
        {
            kx[i] = i;
            ky[i] = i;
        }
        //------

        double L = 10;
        double gamma = 5. / 3;
        // double hrms = 3E-4 * L;
        // double H = Kokkos::sqrt((1. / 4) * (ak * ak + bk * bk + ck * ck + dk * dk)) / hrms;

        auto const x_d = m_grid.x;
        auto const y_d = m_grid.y;
        auto const z_d = m_grid.z;
        auto const& gravity = m_gravity;
        auto const& param_setup = m_param_setup;

        Kokkos::parallel_for(
            "Rayleigh_Taylor_3D_init",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                double P0 = (2 * units::pi * (param_setup.rho0 + param_setup.rho1)
                    * Kokkos::fabs(gravity(i, j, k, 2)) * L) * units::pressure;

                double x = x_d(i) * units::m;
                double y = y_d(j) * units::m;
                double z = z_d(k) * units::m;

                double X = 2 * units::pi * x / L;
                double Y = 2 * units::pi * y / L;
                double h = 0;

                for (int ik = 0; ik < kx.size(); ++ik)
                {
                    for (int jk = 0; jk < ky.size(); ++jk)
                    {
                        double K = kx[ik] * kx[ik] + ky[jk] + ky[jk];
                        if (K >= 8 && K <= 16)
                        {
                            h += (ak * Kokkos::cos(kx[ik] * X) * Kokkos::cos(ky[jk] * Y)
                        + bk * Kokkos::cos(kx[ik] * X) * Kokkos::sin(ky[jk] * Y)
                        + ck * Kokkos::sin(kx[ik] * X) * Kokkos::cos(ky[jk] * Y)
                        + dk * Kokkos::sin(kx[ik] * X) * Kokkos::sin(ky[jk] * Y));
                        }
                    }
                }

                if(z >= h)
                {
                    rho(i, j, k) = param_setup.rho0 * Kokkos::pow(1 - (gamma - 1) / gamma
                                * (param_setup.rho0 * Kokkos::fabs(gravity(i, j, k, 2)) * z) / P0, 1. / (gamma - 1)) * units::density;

                    P(i, j, k) = P0 * Kokkos::pow(rho(i, j, k) / param_setup.rho0, gamma) * units::pressure;

                    fx(i, j, k, 0) = 1;
                }

                if(z < h)
                {
                    rho(i, j, k) = param_setup.rho1 * Kokkos::pow(1 - (gamma - 1) / gamma
                                * (param_setup.rho1 * Kokkos::fabs(gravity(i, j, k, 2)) * z) / P0, 1. / (gamma - 1)) * units::density;

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
