
#pragma once

#include <Kokkos_Core.hpp>
#include <units.hpp>

#include <inih/INIReader.hpp>

#include "default_boundary_setup.hpp"
#include "default_shift_criterion.hpp"
#include "default_user_step.hpp"
#include "eos.hpp"
#include <grid.hpp>
#include <grid_type.hpp>
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
    double fx0;
    double fx1;
    double inter;

    explicit ParamSetup(INIReader const& reader)
        : rho0(reader.GetReal("Initialisation", "rho0", 1.0))
        , rho1(reader.GetReal("Initialisation", "rho1", 1.0))
        , u0(reader.GetReal("Initialisation", "u0", 1.0))
        , u1(reader.GetReal("Initialisation", "u1", 1.0))
        , P0(reader.GetReal("Initialisation", "P0", 1.0))
        , P1(reader.GetReal("Initialisation", "P1", 1.0))
        , fx0(reader.GetReal("Initialisation", "fx0", 1.0))
        , fx1(reader.GetReal("Initialisation", "fx1", 1.0))
        , inter(reader.GetReal("Initialisation", "inter", 1.0))
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
        KV_double_4d const& fx) const final
    {
        assert(equal_extents({0, 1, 2}, rho, u, P, fx));
        assert(u.extent_int(3) == ndim);

        auto const xc = grid.x_center;
        auto const& param_setup = m_param_setup;

        Kokkos::parallel_for(
            "shock_tube_init",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                if(xc(i) * units::m <= param_setup.inter)
                {
                    rho(i, j, k) = param_setup.rho0 * units::density;
                    P(i, j, k) = param_setup.P0 * units::pressure;
                    for (int idim = 0; idim < ndim; ++idim)
                    {
                        u(i, j, k, idim) = param_setup.u0 * units::velocity;
                    }
                        fx(i, j, k, 0) = param_setup.fx0;
                }
                else
                {
                    rho(i, j, k) = param_setup.rho1 * units::density;
                    P(i, j, k) = param_setup.P1 * units::pressure;
                    for (int idim = 0; idim < ndim; ++idim)
                    {
                        u(i, j, k, idim) = param_setup.u1 * units::velocity;
                    }
                        fx(i, j, k, 0) = param_setup.fx1;
                }
            });
    }
};

class GridSetup : public IGridType
{
private :
    Param m_param;

public:
    explicit GridSetup(
        Param param)
        : m_param(std::move(param))
    {
    }

    void execute(
        std::array<int, 3> const Nghost,
        std::array<int, 3> const Nx_glob_ng,
        KVH_double_1d const& x_glob,
        KVH_double_1d const& y_glob,
        KVH_double_1d const& z_glob) const final
    {
        double const Lx = m_param.xmax - m_param.xmin;
        double const dx = Lx / Nx_glob_ng[0];
        x_glob(Nghost[0]) = m_param.xmin;

        int const quarter_x = Nx_glob_ng[0] / 4;
        int const three_quarters_x = 3 * quarter_x;

        for (int i = Nghost[0]+1; i < x_glob.extent_int(0) ; ++i)
        {
            double dxloc;
            if ((i >= quarter_x) && (i <= three_quarters_x))
            {
                dxloc = dx / 5;
            }
            else
            {
                dxloc = dx / 4;
            }
            x_glob(i) = x_glob(i-1) + dxloc;
        }

        double const val_xmax = x_glob(x_glob.extent_int(0)-1 - Nghost[0]);
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
        double const Ly = m_param.ymax - m_param.ymin;
        double const dy = Ly / Nx_glob_ng[1];

        double const Lz = m_param.zmax - m_param.zmin;
        double const dz = Lz / Nx_glob_ng[2];

        for (int i = 0; i < y_glob.extent_int(0) ; ++i)
        {
            y_glob(i) = m_param.ymin + i * dy;
            z_glob(i) = m_param.zmin + i * dz;
        }
    }
};

} // namespace novapp
