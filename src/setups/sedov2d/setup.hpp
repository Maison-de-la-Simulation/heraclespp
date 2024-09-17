
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
#include <geom.hpp>
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
    double P0;
    double E0;
    double E1;
    double nx;
    double ny;

    explicit ParamSetup(INIReader const& reader)
        : rho0(reader.GetReal("Initialisation", "rho0", 1.0))
        , u0(reader.GetReal("Initialisation", "u0", 1.0))
        , P0(reader.GetReal("Initialisation", "P0", 1.0))
        , E0(reader.GetReal("Initialisation", "E0", 1.0))
        , E1(reader.GetReal("Initialisation", "E1", 1.0))
        , nx(reader.GetReal("Grid", "Nx_glob", 1.0))
        , ny(reader.GetReal("Grid", "Ny_glob", 1.0))
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
        auto const& dv = grid.dv;
        auto const& eos = m_eos;
        auto const& param_setup = m_param_setup;

        Kokkos::parallel_for(
            "Sedov_2D_init",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                if (geom == Geometry::Geom_cartesian)
                {
                    double const x = x_d(i) * units::m;
                    double const y = y_d(j) * units::m;
                    double const r = Kokkos::sqrt(x * x + y * y);

                    rho(i, j, k) = param_setup.rho0 * units::density;
                    for (int idim = 0; idim < ndim; ++idim)
                    {
                        u(i, j, k, idim) = param_setup.u0 * units::velocity;
                    }

                    if (r < 0.025)
                    {
                        double const evol = param_setup.E1 / dv(i, j, k) * units::evol;
                        double const T = eos.compute_T_from_evol(rho(i, j, k), evol) * units::Kelvin;
                        P(i, j, k) = eos.compute_P_from_T(rho(i, j, k), T) * units::pressure;
                    }
                    else
                    {
                        double const evol = param_setup.E0 / dv(i, j, k)  * units::evol;
                        double const T = eos.compute_T_from_evol(rho(i, j, k), evol) * units::Kelvin;
                        P(i, j, k) = eos.compute_P_from_T(rho(i, j, k), T) * units::pressure;
                    }
                }
            });
    }
};

} // namespace novapp
