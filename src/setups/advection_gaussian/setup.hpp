
#pragma once

#include <Kokkos_Core.hpp>
#include <units.hpp>

#include <inih/INIReader.hpp>

#include "eos.hpp"
#include "../../mesh/grid.hpp"
#include "../../mesh/grid_type.hpp"
#include "initialization_interface.hpp"
#include "kokkos_shortcut.hpp"
#include "ndim.hpp"
#include "nova_params.hpp"
#include "../../mesh/range.hpp"

namespace novapp
{

class ParamSetup
{
public:
    double u0;
    double P0;

    explicit ParamSetup(INIReader const& reader)
        : u0(reader.GetReal("Initialisation", "u0", 1.0))
        , P0(reader.GetReal("Initialisation", "P0", 1.0))
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
        Gravity const& gravity)
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

        auto const xc = m_grid.x_center;

        auto const [begin, end] = cell_range(range);
        Kokkos::parallel_for(
            "advection_gaussian_init",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
            KOKKOS_CLASS_LAMBDA(int i, int j, int k)
            {
                rho(i, j, k) = 1 * Kokkos::exp(- 15 * Kokkos::pow(1. / 2 - xc(i), 2)) * units::density;

                for (int idim = 0; idim < ndim; ++idim)
                {
                    u(i, j, k, idim) = m_param_setup.u0 * units::velocity;
                }

                P(i, j, k) = m_param_setup.P0 * units::pressure;
            });
    }
};

class GridSetup : public IGridType
{
public:
    explicit GridSetup(
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
