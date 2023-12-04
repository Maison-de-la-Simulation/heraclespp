
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
    double u0;

    explicit ParamSetup(INIReader const& reader)
        : u0(reader.GetReal("Initialisation", "u0", 1.0))
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

        auto const x = m_grid.x;
        auto const y = m_grid.y;
        auto const z = m_grid.z;
        auto const& param_setup = m_param_setup;

        auto const [begin, end] = cell_range(range);
        Kokkos::parallel_for(
            "Rayleigh_Taylor_3D_sph_init",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                double R = 1.5;
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<double> dist(-1.0, 1.0);
                double nr = dist(rd);

                if( x(i) < R)
                {
                    rho(i, j, k) = 1. / (Kokkos::numbers::pi * R * R); // * (1 + 0.01 * nr);
                    for (int idim = 0; idim < ndim; ++idim)
                    {
                        u(i, j, k, idim) =  m_param_setup.u0 * x(i) / R; // * (1 + 0.01 * nr);
                    }
                }
                else
                {
                    rho(i, j, k) = 1 ;// * (1 + 0.01 * nr);
                    for (int idim = 0; idim < ndim; ++idim)
                    {
                        u(i, j, k, idim) = 0;
                    }
                }
                P(i, j, k) = std::pow(10, -5) * rho(i, j, k) *  m_param_setup.u0 * m_param_setup.u0;
            });
    }
};

} // namespace novapp
