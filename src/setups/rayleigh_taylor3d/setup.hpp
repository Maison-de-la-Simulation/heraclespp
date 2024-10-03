
#pragma once

#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <units.hpp>
#include <numeric>
#include <random>

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
    int nx;
    int ny;

    explicit ParamSetup(INIReader const& reader)
        : rho0(reader.GetReal("Initialisation", "rho0", 1.0))
        , rho1(reader.GetReal("Initialisation", "rho1", 1.0))
        , u0(reader.GetReal("Initialisation", "u0", 1.0))
        , P0(reader.GetReal("Initialisation", "P0", 1.0))
        , nx(reader.GetInteger("Grid", "Nx_glob", 0))
        , ny(reader.GetInteger("Grid", "Ny_glob", 0))
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

        int const mpi_rank = grid.mpi_rank;
        MPI_Comm const comm_cart = grid.comm_cart;

        std::array<double, 4> data_to_broadcast;

        if (mpi_rank == 0)
        {
            std::random_device rd;
            //std::mt19937 const gen(rd());
            std::mt19937 gen(42);
            std::uniform_real_distribution<double> dist(-1.0, 1.0);

            for(double& data : data_to_broadcast)
            {
                //data = dist(rd);
                data = dist(gen);
            }
        }

        MPI_Bcast(data_to_broadcast.data(), 4, MPI_DOUBLE, 0, comm_cart);

        double const ak = data_to_broadcast[0];
        double const bk = data_to_broadcast[1];
        double const ck = data_to_broadcast[2];
        double const dk = data_to_broadcast[3];

        std::cout << "[MPI process "<< mpi_rank <<"] ak = " << ak << std::endl;
        std::cout << "[MPI process "<< mpi_rank <<"] bk = " << bk << std::endl;
        std::cout << "[MPI process "<< mpi_rank <<"] ck = " << ck << std::endl;
        std::cout << "[MPI process "<< mpi_rank <<"] dk = " << dk << std::endl;

        Kokkos::Array<int, 5> kx;
        std::iota(kx.data(), kx.data() + 5, 0);
        Kokkos::Array<int, 5> ky;
        std::iota(ky.data(), ky.data() + 5, 0);

        double const L = 10;
        double const gamma = 5. / 3;
        double hrms = 3E-4 * L;
        double H_norm = Kokkos::sqrt((1. / 4) * (ak * ak + bk * bk + ck * ck + dk * dk)) / hrms;

        auto const x = grid.x;
        auto const y = grid.y;
        auto const z = grid.z;
        auto const& gravity = m_gravity;
        auto const& param_setup = m_param_setup;
        int const nghost_x = grid.Nghost[0];
        int const nghost_y = grid.Nghost[1];

        Kokkos::View<double**> interface("interface_RT", m_param_setup.nx, m_param_setup.ny);

        Kokkos::parallel_for(
            "interface",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {param_setup.nx, param_setup.ny}),
            KOKKOS_LAMBDA(int i, int j)
            {
                int i_offset = i + nghost_x;
                int j_offset = j + nghost_y;

                double X = 2 * units::pi * x(i_offset) / L;
                double Y = 2 * units::pi * y(j_offset) / L;
                double h = 0;

                for (std::size_t ik = 0; ik < Kokkos::Array<int, 5>::size(); ++ik) // loop kx
                {
                    for (std::size_t jk = 0; jk < Kokkos::Array<int, 5>::size(); ++jk) // loop ky
                    {
                        double K_test = kx[ik] * kx[ik] + ky[jk] * ky[jk];
                        if (K_test >= 8 && K_test <= 16)
                        {
                            h += (ak * Kokkos::cos(kx[ik] * X) * Kokkos::cos(ky[jk] * Y)
                                + bk * Kokkos::cos(kx[ik] * X) * Kokkos::sin(ky[jk] * Y)
                                + ck * Kokkos::sin(kx[ik] * X) * Kokkos::cos(ky[jk] * Y)
                                + dk * Kokkos::sin(kx[ik] * X) * Kokkos::sin(ky[jk] * Y));
                        }
                    }
                }

                interface(i, j) = h / H_norm;
                //std::cout << "h = " << interface(i, j) << std::endl;
            });

        Kokkos::parallel_for(
            "Rayleigh_Taylor_3D_init",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                double P0 = (2 * units::pi * (param_setup.rho0 + param_setup.rho1)
                            * Kokkos::fabs(gravity(i, j, k, 2)) * L);

                for (int idim = 0; idim < ndim; ++idim)
                {
                    u(i, j, k, idim) = param_setup.u0;
                }

                if(z(k) >= interface(i, j))
                {
                    rho(i, j, k) = param_setup.rho0 * Kokkos::pow(1 - (gamma - 1) / gamma
                                * (param_setup.rho0 * Kokkos::fabs(gravity(i, j, k, 2)) * z(k)) / P0, 1. / (gamma - 1));
                    P(i, j, k) = P0 * Kokkos::pow(rho(i, j, k) / param_setup.rho0, gamma);
                    fx(i, j, k, 0) = 1;
                }
                else
                {
                    rho(i, j, k) = param_setup.rho1 * Kokkos::pow(1 - (gamma - 1) / gamma
                                * (param_setup.rho1 * Kokkos::fabs(gravity(i, j, k, 2)) * z(k)) / P0, 1. / (gamma - 1));
                    P(i, j, k) = P0 * Kokkos::pow(rho(i, j, k) / param_setup.rho1, gamma);
                    fx(i, j, k, 0) = 0;
                }

                 std:: cout << i << " " << j << " " << k <<" " << z(k) << " " << interface(i, j) << " " << fx(i, j, k, 0) << std::endl;
            });

        /* Kokkos::parallel_for(
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

                for (std::size_t ik = 0; ik < Kokkos::Array<int, 5>::size(); ++ik)
                {
                    for (std::size_t jk = 0; jk < Kokkos::Array<int, 5>::size(); ++jk)
                    {
                        double const K = kx[ik] * kx[ik] + ky[jk] * ky[jk];
                        if (K >= 8 && K <= 16)
                        {
                            h += (ak * Kokkos::cos(kx[ik] * X) * Kokkos::cos(ky[jk] * Y)
                        + bk * Kokkos::cos(kx[ik] * X) * Kokkos::sin(ky[jk] * Y)
                        + ck * Kokkos::sin(kx[ik] * X) * Kokkos::cos(ky[jk] * Y)
                        + dk * Kokkos::sin(kx[ik] * X) * Kokkos::sin(ky[jk] * Y));
                        }
                    }
                }

                if(z >= (1 / H) * h)
                {
                    rho(i, j, k) = param_setup.rho0 * Kokkos::pow(1 - (gamma - 1) / gamma
                                * (param_setup.rho0 * Kokkos::fabs(gravity(i, j, k, 2)) * z) / P0, 1. / (gamma - 1)) * units::density;

                    P(i, j, k) = P0 * Kokkos::pow(rho(i, j, k) / param_setup.rho0, gamma) * units::pressure;

                    fx(i, j, k, 0) = 1;
                }

                if(z < (1 / H) * h)
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
            }); */
    }
};

} // namespace novapp
