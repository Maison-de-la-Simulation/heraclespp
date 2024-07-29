
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
    double u0;
    double P0;

    explicit ParamSetup(INIReader const& reader)
        : rho0(reader.GetReal("Initialisation", "rho0", 1.0))
        , u0(reader.GetReal("Initialisation", "u0", 1.0))
        , P0(reader.GetReal("Initialisation", "P0", 1.0))
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
        [[maybe_unused]] Grid const& grid,
        KV_double_3d const& rho,
        KV_double_4d const& u,
        KV_double_3d const& P,
        [[maybe_unused]] KV_double_4d const& fx) const final
    {
        assert(equal_extents({0, 1, 2}, rho, u, P, fx));
        assert(u.extent_int(3) == ndim);

        auto const& param_setup = m_param_setup;

        Kokkos::parallel_for(
            "shock_wall_init",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                rho(i, j, k) = param_setup.rho0 * units::density;

                P(i, j, k) = param_setup.P0 * units::pressure;

                for (int idim = 0; idim < ndim; ++idim)
                {
                    u(i, j, k, idim) = param_setup.u0 * units::velocity;
                }

                // double T = m_eos.compute_T_from_P(rho(i, j, k), P(i, j, k));
                // double Pr = units::ar * T * T * T * T / 3;
                // double Pg = rho(i, j, k)  * units::kb * T / (1 * units::mp);

                //std::cout<<"Pg = "<<Pg<<" Pr = "<<Pr<<" alpha = "<< Pr/Pg<<std::endl;
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
        // regular grid
    }

    void execute(
        std::array<int, 3> Nghost,
        std::array<int, 3> Nx_glob_ng,
        KVH_double_1d const& x_glob,
        KVH_double_1d const& y_glob,
        KVH_double_1d const& z_glob) const final
    {
        double dx = m_param.xmax / (2 * Nx_glob_ng[0]);
        x_glob(Nghost[0]) = 0;
        for (int i = Nghost[0]+1; i < x_glob.extent_int(0) ; ++i)
        {
            x_glob(i) = x_glob(i-1) + dx;
            dx *= 1.005;
        }

        double val_xmax = x_glob(x_glob.extent_int(0)-1 - Nghost[0]);
        for (int i = Nghost[0]; i < x_glob.extent_int(0); ++i)
        {
            x_glob(i) = m_param.xmax * x_glob(i) / val_xmax;
        }

        // Reflexive X-left ghost cells
        for(int i = Nghost[0]-1; i >= 0; i--)
        {
            int mirror = Nghost[0] -  2 * i + 1;
            x_glob(i) = x_glob(i+1) - (x_glob(i+mirror+1) - x_glob(i+mirror));
        }

        // Y and Z
        double Ly = m_param.ymax - m_param.ymin;
        double dy = Ly / Nx_glob_ng[1];

        double Lz = m_param.zmax - m_param.zmin;
        double dz = Lz / Nx_glob_ng[2];

        for (int i = 0; i < y_glob.extent_int(0) ; ++i)
        {
            y_glob(i) = m_param.ymin + i * dy;
            z_glob(i) = m_param.zmin + i * dz;
        }
    }
};

} // namespace novapp
