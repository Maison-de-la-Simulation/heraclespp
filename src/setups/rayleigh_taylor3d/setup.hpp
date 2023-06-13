
#pragma once

#include <random>

#include <Kokkos_Core.hpp>
#include <PerfectGas.hpp>

#include <inih/INIReader.hpp>

#include "ndim.hpp"
#include "range.hpp"
#include "Kokkos_shortcut.hpp"
#include "grid.hpp"
#include "units.hpp"
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

class InitializationSetup : public IInitializationProblem
{
private:
    thermodynamics::PerfectGas m_eos;
    Grid m_grid;
    ParamSetup m_param_setup;

public:
    InitializationSetup(
        thermodynamics::PerfectGas const& eos,
        Grid const& grid,
        ParamSetup const& param_set_up)
        : m_eos(eos)
        , m_grid(grid)
        , m_param_setup(param_set_up)
    {
    }

    void execute(
        Range const& range,
        KV_double_3d const rho,
        KV_double_4d const u,
        KV_double_3d const P,
        KV_double_4d const fx,
        KV_double_1d g) const final
    {
        assert(rho.extent(0) == u.extent(0));
        assert(u.extent(0) == P.extent(0));
        assert(rho.extent(1) == u.extent(1));
        assert(u.extent(1) == P.extent(1));
        assert(rho.extent(2) == u.extent(2));
        assert(u.extent(2) == P.extent(2));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        double ak = dist(rd);
        double bk = dist(rd);
        double ck = dist(rd);
        double dk = dist(rd);
        /* std::cout << "ak: " << ak << std::endl;
        std::cout << "bk: " << bk << std::endl;
        std::cout << "ck: " << ck << std::endl;
        std::cout << "dk: " << dk << std::endl; */

        std::array<int, 5> kx;
        std::array<int, 5> ky;
        for (int i = 0; i < kx.size(); ++i)
        {
            kx[i] = i;
            ky[i] = i;
        }
        //------

        double L = 10;
        double P0 = (2 * units::pi * (m_param_setup.rho0 + m_param_setup.rho1)
                   * std::abs(g(2)) * L) * units::pressure;
        double gamma = 5. / 3;
        double hrms = 3E-4 * L;
        double H = std::sqrt((1. / 4) * (ak * ak + bk * bk + ck * ck + dk * dk)) / hrms;

        auto const x_d = m_grid.x.d_view;
        auto const y_d = m_grid.y.d_view;
        auto const z_d = m_grid.z.d_view;
        auto const [begin, end] = cell_range(range);
        Kokkos::parallel_for(
        "Rayleigh_Taylor3d_init",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
        KOKKOS_CLASS_LAMBDA(int i, int j, int k)
        {
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
                rho(i, j, k) = m_param_setup.rho0 * std::pow(1 - (gamma - 1) / gamma 
                              * (m_param_setup.rho0 * std::abs(g(2)) * z) / P0, 1. / (gamma - 1)) * units::density;
                P(i, j, k) = P0 * std::pow(rho(i, j, k) / m_param_setup.rho0, gamma) * units::pressure;
                fx(i, j, k, 0) = 1;
            }
            if(z < h)
            {
                rho(i, j, k) = m_param_setup.rho1 * std::pow(1 - (gamma - 1) / gamma 
                              * (m_param_setup.rho1 * std::abs(g(2)) * z) / P0, 1. / (gamma - 1)) * units::density;
                P(i, j, k) = P0 * std::pow(rho(i, j, k) / m_param_setup.rho1, gamma) * units::pressure;
                fx(i, j, k, 0) = 0;
            }
            for (int idim = 0; idim < ndim; ++idim)
            {
                u(i, j, k, idim) = m_param_setup.u0 * units::velocity;
            }
        });
    }
};

class BoundarySetup : public IBoundaryCondition
{
public:
    BoundarySetup(int idim, int iface,
        [[maybe_unused]] thermodynamics::PerfectGas const& eos,
        [[maybe_unused]] Grid const& grid,
        [[maybe_unused]] ParamSetup const& param_setup)
        : IBoundaryCondition(idim, iface)
    {
    }
};

} // namespace novapp
