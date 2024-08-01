
#pragma once

#include <Kokkos_Core.hpp>
#include <units.hpp>

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
    ParamSetup m_param_setup;

public:
    InitializationSetup(
        EOS const& eos,
        ParamSetup const& param_set_up,
        [[maybe_unused]] Gravity const& gravity)
        : m_eos(eos)
        , m_param_setup(param_set_up)
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

        auto const x_d = grid.x;
        auto const y_d = grid.y;
        auto const& eos = m_eos;
        auto const& param_setup = m_param_setup;

        Kokkos::parallel_for(
            "Gresho_vortex_init",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                double x = x_d(i) * units::m;
                double y = y_d(j) * units::m;
                double r = Kokkos::sqrt(x * x + y * y);
                double theta = Kokkos::atan2(y, x);
                double u_theta;
                double P0 = 1. / (eos.adiabatic_index() * param_setup.Ma * param_setup.Ma);

                rho(i, j, k) = param_setup.rho0 * units::density;

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
                    u(i, j, k, idim) = param_setup.u0 * units::velocity;
                    }

                    P(i, j, k) = P0 * units::pressure - 2 + 4 * Kokkos::log(2);
                }
            });
    }
};

} // namespace novapp
