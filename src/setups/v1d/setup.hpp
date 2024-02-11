#pragma once

#include <Kokkos_Core.hpp>
#include <units.hpp>

#include <inih/INIReader.hpp>

#include "default_boundary_setup.hpp"
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
    explicit ParamSetup(INIReader const& reader)
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
        KV_double_4d const fx) const final
    {
        assert(rho.extent(0) == u.extent(0));
        assert(u.extent(0) == P.extent(0));
        assert(rho.extent(1) == u.extent(1));
        assert(u.extent(1) == P.extent(1));
        assert(rho.extent(2) == u.extent(2));
        assert(u.extent(2) == P.extent(2));

        Kokkos::parallel_for(
            "1d_to_3d_test_init",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                //
            });
    }
};

class GridSetup : public IGridType
{
private :
    Param m_param;

public:
    explicit GridSetup(Param const& param)
        : m_param(param)
    {
    }

    void execute(
        std::array<int, 3> Nghost,
        std::array<int, 3> Nx_glob_ng,
        KVH_double_1d x_glob,
        KVH_double_1d y_glob,
        KVH_double_1d z_glob) const final
    {
        if (ndim == 3)
        {
             auto const& param = m_param;

            //compute_regular_mesh_1d(y_glob, Nghost[1], param.ymin, (param.ymax - param.ymin) / Nx_glob_ng[1]);
            //compute_regular_mesh_1d(z_glob, Nghost[2], param.zmin, (param.zmax - param.zmin) / Nx_glob_ng[2]);

            double dy = (param.ymax - param.ymin) / Nx_glob_ng[1];
            y_glob(Nghost[1]) = param.ymin;
            for (int i = Nghost[1]+1; i < y_glob.extent_int(0); i++)
            {
                y_glob(i) = y_glob(i-1) + dy;
            }

            double dz = (param.zmax - param.zmin) / Nx_glob_ng[2];
            z_glob(Nghost[2]) = param.zmin;
            for (int i = Nghost[2]+1; i < z_glob.extent_int(0) ; i++)
            {
                z_glob(i) = z_glob(i-1) + dz;
            }

            // Left ghost cells
            for(int i = Nghost[1]-1; i >= 0; i--)
            {
                y_glob(i) = y_glob(i+1) - dy;
            }
            for(int i = Nghost[2]-1; i >= 0; i--)
            {
                z_glob(i) = z_glob(i+1) - dz;
            }
        }
        else
        {}
    }
};

} // namespace novapp
