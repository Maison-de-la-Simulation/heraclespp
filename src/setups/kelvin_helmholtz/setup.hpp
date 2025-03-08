
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
    double P0;

    explicit ParamSetup(INIReader const& reader)
        : P0(reader.GetReal("Initialisation", "P0", 1.0))
    {
    }
};

template <class Gravity>
class InitializationSetup : public IInitializationProblem
{
private:
    thermodynamics::PerfectGas m_eos;
    ParamSetup m_param_setup;

public:
    InitializationSetup(
        thermodynamics::PerfectGas const& eos,
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

        auto const xc = grid.x_center;
        auto const yc = grid.y_center;
        auto const& param_setup = m_param_setup;

        double const drho_rho = 1;
        double const a = 0.05;
        double const amp = 0.01;
        double const sigma = 0.2;
        double const y1 = 0.5;
        double const y2 = 1.5;

        Kokkos::parallel_for(
            "Kelvin_Helmholtz_init",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                double const x = xc(i);
                double const y = yc(j);

                rho(i, j, k)  = 1 + drho_rho * (Kokkos::tanh((y - y1) / a) - Kokkos::tanh((y - y2) / a));

                u(i, j, k, 0) = 1 * (Kokkos::tanh((y - y1) / a) - Kokkos::tanh((y - y2) / a));

                u(i, j, k, 1) = amp * Kokkos::sin(Kokkos::numbers::pi * x)
                                * (Kokkos::exp(- (y - y1) * (y - y1) / (sigma * sigma))
                                + Kokkos::exp(- (y - y2) * (y - y2) / (sigma * sigma)));

                P(i, j, k) = param_setup.P0;
            });
    }
};

} // namespace novapp
