
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
    double Ma;

    explicit ParamSetup(INIReader const& reader)
        : rho0(reader.GetReal("Initialisation", "rho0", 1.0))
        , u0(reader.GetReal("Initialisation", "u0", 1.0))
        , Ma(reader.GetReal("Initialisation", "Ma", 1.0))
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
            "Gresho_vortex_init",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
            KOKKOS_CLASS_LAMBDA(int i, int j, int k)
            {
                double x = x_d(i) * units::m;
                double y = y_d(j) * units::m;
                double r = Kokkos::sqrt(x * x + y * y);
                double theta = Kokkos::atan2(y, x);
                double u_theta;
                double P0 = 1. / (m_eos.adiabatic_index() * m_param_setup.Ma * m_param_setup.Ma);
                
                rho(i, j, k) = m_param_setup.rho0 * units::density;

                if (r < 0.2)
                {
                    u_theta = 5 * r;
                    u(i, j, k, 0) = - u_theta * Kokkos::sin(theta);
                    u(i, j, k, 1) = u_theta * Kokkos::cos(theta);

                    P(i, j, k) = P0 * units::pressure + 12.5 * r * r;
                }

                else if ((r >= 0.2) && (r < 0.4))
                {
                    u_theta = 2 - 5 * r;
                    u(i, j, k, 0) = - u_theta * Kokkos::sin(theta);
                    u(i, j, k, 1) = u_theta * Kokkos::cos(theta);

                    P(i, j, k) = P0 * units::pressure + 12.5 * r * r + 4 - 20 * r + 4 * Kokkos::log(5 * r);
                }

                else
                {
                    for (int idim = 0; idim < ndim; ++idim)
                    {
                    u(i, j, k, idim) = m_param_setup.u0 * units::velocity;
                    }

                    P(i, j, k) = P0 * units::pressure - 2 + 4 * Kokkos::log(2);
                }
            });
    }
};

} // namespace novapp
