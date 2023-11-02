
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
    Grid m_grid;
    ParamSetup m_param_setup;

public:
    InitializationSetup(
        EOS const& eos,
        Grid const& grid,
        ParamSetup const& param_set_up,
        [[maybe_unused]] Gravity const& gravity)
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
        [[maybe_unused]] KV_double_4d const fx) const final
    {
        assert(rho.extent(0) == u.extent(0));
        assert(u.extent(0) == P.extent(0));
        assert(rho.extent(1) == u.extent(1));
        assert(u.extent(1) == P.extent(1));
        assert(rho.extent(2) == u.extent(2));
        assert(u.extent(2) == P.extent(2));

        auto const x_d = m_grid.x;
        auto const y_d = m_grid.y;

        auto const [begin, end] = cell_range(range);
        Kokkos::parallel_for(
            "Sedov_2D_init",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
            KOKKOS_CLASS_LAMBDA(int i, int j, int k)
            {
                double x = x_d(i) * units::m;
                double y = y_d(j) * units::m;
                double r = Kokkos::sqrt(x * x + y * y);

                rho(i, j, k) = m_param_setup.rho0 * units::density;

                for (int idim = 0; idim < ndim; ++idim)
                {
                    u(i, j, k, idim) = m_param_setup.u0 * units::velocity;
                }

                if (r <0.025)
                {
                    double T = m_eos.compute_T_from_evol(rho(i, j, k), 
                                m_param_setup.E1 * units::evol / m_grid.dv(i, j, k)) * units::Kelvin;
                    P(i, j, k) = m_eos.compute_P_from_T(rho(i, j, k), T) * units::pressure;
                }
                
                else
                {
                    double T = m_eos.compute_T_from_evol(rho(i, j, k), 
                                m_param_setup.E0 * units::evol / m_grid.dv(i, j, k)) * units::Kelvin;
                    P(i, j, k) = m_eos.compute_P_from_T(rho(i, j, k), T) * units::pressure;
                }
            });
    }
};

} // namespace novapp
