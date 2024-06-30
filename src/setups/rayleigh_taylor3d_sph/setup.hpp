
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

#include <Kokkos_Random.hpp>

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
    ParamSetup m_param_setup;
    Gravity m_gravity;

public:
    InitializationSetup(
        thermodynamics::PerfectGas const& eos,
        ParamSetup const& param_set_up,
        Gravity const& gravity)
        : m_eos(eos)
        , m_param_setup(param_set_up)
        , m_gravity(gravity)
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
        assert(rho.extent(0) == u.extent(0));
        assert(u.extent(0) == P.extent(0));
        assert(rho.extent(1) == u.extent(1));
        assert(u.extent(1) == P.extent(1));
        assert(rho.extent(2) == u.extent(2));
        assert(u.extent(2) == P.extent(2));

        auto const x = grid.x;
        auto const y = grid.y;
        auto const z = grid.z;
        auto const& param_setup = m_param_setup;
        auto const& eos = m_eos;

        double R = 0.1;
        Kokkos::Random_XorShift64_Pool<> random_pool(12345 + grid.mpi_rank);

        Kokkos::parallel_for(
            "Rayleigh_Taylor_3D_sph_init",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                auto generator = random_pool.get_state();
                double random_number = generator.drand(-1.0, 1.0);
                double perturb = (1 + 0.01 * random_number);
                random_pool.free_state(generator);

                double eint;

                if( x(i) < R)
                {
                    rho(i, j, k) = 10 * perturb;
                    u(i, j, k, 0) = param_setup.u0 * x(i) / R * perturb;
                    u(i, j, k, 1) = 0;
                    u(i, j, k, 2) = 0;
                    eint = 100;
                }
                else
                {
                    rho(i, j, k) = 0.1 * perturb;
                    for (int idim = 0; idim < ndim; ++idim)
                    {
                        u(i, j, k, idim) = 0;
                    }
                    eint = 0.01;
                }
                P(i, j, k) = eos.compute_P_from_evol(rho(i, j, k), eint);
            });
    }
};

} // namespace novapp
