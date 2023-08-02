
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
    double u1;
    double P0;
    double P1;
    double fx0;
    double fx1;

    explicit ParamSetup(INIReader const& reader)
        : rho0(reader.GetReal("Initialisation", "rho0", 1.0))
        , rho1(reader.GetReal("Initialisation", "rho1", 1.0))
        , u0(reader.GetReal("Initialisation", "u0", 1.0))
        , u1(reader.GetReal("Initialisation", "u1", 1.0))
        , P0(reader.GetReal("Initialisation", "P0", 1.0))
        , P1(reader.GetReal("Initialisation", "P1", 1.0))
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
        KV_double_4d const fx) const final
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
            "shock_tube_init",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
            KOKKOS_CLASS_LAMBDA(int i, int j, int k)
            {
                if(xc(i) * units::m <= 0.5)
                {
                    rho(i, j, k) = m_param_setup.rho0 * units::density;
                    P(i, j, k) = m_param_setup.P0 * units::pressure;
                    for (int idim = 0; idim < ndim; ++idim)
                    {
                        u(i, j, k, idim) = m_param_setup.u0 * units::velocity;
                    }
                    fx(i, j, k, 0) = m_param_setup.fx0;
                }

                else
                {
                    rho(i, j, k) = m_param_setup.rho1 * units::density;
                    P(i, j, k) = m_param_setup.P1 * units::pressure;
                    for (int idim = 0; idim < ndim; ++idim)
                    {
                        u(i, j, k, idim) = m_param_setup.u1 * units::velocity;
                    }
                    fx(i, j, k, 0) = m_param_setup.fx1;
                }
            });
    }
};

class GridSetup : public IGridType
{
private :
    Param m_param;

public:
    GridSetup(
        Param const& param)
        : IGridType()
        , m_param(param)
    {
    }

    void execute(
        KVH_double_1d x_glob,
        [[maybe_unused]] KVH_double_1d y_glob,
        [[maybe_unused]] KVH_double_1d z_glob,
        std::array<int, 3> Nghost,
        std::array<int, 3> Nx_glob_ng) const final
    {
        double Lx = m_param.xmax - m_param.xmin;
        double dx = Lx / Nx_glob_ng[0];
        x_glob(Nghost[0]) = m_param.xmin;

        int quater_x = Nx_glob_ng[0] / 4;
        int three_quaters_x = 3 * quater_x;

        for (int i = Nghost[0]+1; i < x_glob.extent_int(0) ; i++)
        {
            double dxloc;
            if ((i >= quater_x) && (i <= three_quaters_x))
            {
                dxloc = dx / 5;
            }
            else
            {
                dxloc = dx / 4;
            }
            x_glob(i) = x_glob(i-1) + dxloc;
        }

        double val_xmax = x_glob(x_glob.extent_int(0)-1 - Nghost[0]);
        for (int i = Nghost[0]; i < x_glob.extent_int(0); ++i)
        {
            x_glob(i) = m_param.xmax * x_glob(i) / val_xmax;
        }

        // Left ghost cells
        for(int i = Nghost[0]-1; i >= 0; i--)
        {
            x_glob(i) = x_glob(i+1) - dx / 4;
        }

        // Y and Z
        double Ly = m_param.ymax - m_param.ymin;
        double dy = Ly / Nx_glob_ng[1];

        double Lz = m_param.zmax - m_param.zmin;
        double dz = Lz / Nx_glob_ng[2];

        for (int i = 0; i < y_glob.extent_int(0) ; i++)
        {
            y_glob(i) = m_param.ymin + i * dy;
            z_glob(i) = m_param.zmin + i * dz;
        }
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
