
#pragma once

#include <Kokkos_Core.hpp>

#include <inih/INIReader.hpp>

#include "ndim.hpp"
#include "eos.hpp"
#include "range.hpp"
#include "eos.hpp"
#include "kokkos_shortcut.hpp"
#include "grid.hpp"
#include <units.hpp>
#include "initialization_interface.hpp"
#include "nova_params.hpp"

namespace novapp
{

class ParamSetup
{
public:
    double rho0;
    double rho1;
    double u0;
    double P0;
    double A;
    double fx0;
    double fx1;

    explicit ParamSetup(INIReader const& reader)
        : rho0(reader.GetReal("Initialisation", "rho0", 1.0))
        , rho1(reader.GetReal("Initialisation", "rho1", 1.0))
        , u0(reader.GetReal("Initialisation", "u0", 1.0))
        , P0(reader.GetReal("Initialisation", "P0", 1.0))
        , A(reader.GetReal("Initialisation", "A", 1.0))
        , fx0(reader.GetReal("Initialisation", "fx0", 1.0))
        , fx1(reader.GetReal("Initialisation", "fx1", 1.0))
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

        double P0 = (10. / 7 + 1. / 4) * units::pressure;

        auto const x_d = m_grid.x.d_view;
        auto const y_d = m_grid.y.d_view;

        auto const [begin, end] = cell_range(range);
        Kokkos::parallel_for(
            "Rayleigh_Taylor_2D_init",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
            KOKKOS_CLASS_LAMBDA(int i, int j, int k)
            {
                double x = x_d(i) * units::m;
                double y = y_d(j) * units::m;
                double h = 0.01 * Kokkos::cos(4 * Kokkos::numbers::pi * x);

                if (y >= h)
                {
                    rho(i, j, k) = m_param_setup.rho1 * units::density;
                    fx(i, j, k, 0) = m_param_setup.fx0;
                }

                if (y < h)
                {
                    rho(i, j, k) = m_param_setup.rho0 * units::density;
                    fx(i, j, k, 0) = m_param_setup.fx1;
                }
                
                u(i, j, k, 0) = m_param_setup.u0 * units::velocity;
                u(i, j, k, 1) = m_param_setup.u0 * units::velocity;
                /* u(i, j, k, 1) = (m_param_setup.A/4) * (1+Kokkos::cos(2*Kokkos::numbers::pi*x/m_grid.L[0])) 
                                * (1+Kokkos::cos(2*Kokkos::numbers::pi*y/m_grid.L[1])); */
                                
                P(i, j, k) = (P0 + rho(i, j, k) * m_gravity(i, j, k, 1) * units::acc * y) * units::pressure;
            });
    }
};

class GridSetup : public IGridType
{
public:
    GridSetup(
        [[maybe_unused]] Param const& param)
        : IGridType()
    {
        // regular grid
    }
};

template <class Gravity>
class BoundarySetup : public IBoundaryCondition
{
public:
    BoundarySetup(int idim, int iface,
        [[maybe_unused]] EOS const& eos,
        [[maybe_unused]] Grid const& grid,
        [[maybe_unused]] ParamSetup const& param_setup,
        [[maybe_unused]] Gravity const& gravity)
        : IBoundaryCondition(idim, iface)
    {
        // no new boundary
    }
};

} // namespace novapp
