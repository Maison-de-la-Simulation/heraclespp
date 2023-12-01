
#pragma once

#include <Kokkos_Core.hpp>
#include <units.hpp>

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
    double u1;
    double P0;
    double P1;
    explicit ParamSetup(INIReader const& reader)
    : rho0(reader.GetReal("Initialisation", "rho0", 1.0))
    , rho1(reader.GetReal("Initialisation", "rho1", 1.0))
    , u0(reader.GetReal("Initialisation", "u0", 1.0))
    , u1(reader.GetReal("Initialisation", "u1", 1.0))
    , P0(reader.GetReal("Initialisation", "P0", 1.0))
    , P1(reader.GetReal("Initialisation", "P1", 1.0))
    {
    }
};

template <class Gravity>
class InitializationSetup : public IInitializationProblem
{
private:
    EOS m_eos;
    Grid m_grid;
    ParamSetup m_param_setup;
    Gravity m_gravity;

public:
    InitializationSetup(
        EOS const& eos,
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

        auto const [begin, end] = cell_range(range);
        Kokkos::parallel_for(
            "test3d_sph_init",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
            KOKKOS_CLASS_LAMBDA(int i, int j, int k)
            {
                double x = m_grid.x(i) * units::m;
                double R = 1.5;

                if (x < R)
                {
                    rho(i, j, k) =  m_param_setup.rho0;
                    for (int idim = 0; idim < ndim; ++idim)
                    {
                        u(i, j, k, idim) =  m_param_setup.u0;
                    }
                    P(i, j, k) =  m_param_setup.P0;
                }

                if (x >= R)
                {
                    rho(i, j, k) = m_param_setup.rho1;
                    for (int idim = 0; idim < ndim; ++idim)
                    {
                        u(i, j, k, idim) = m_param_setup.u1;
                    }
                    P(i, j, k) = m_param_setup.P1;
                }
            });
    }
};

} // namespace novapp
