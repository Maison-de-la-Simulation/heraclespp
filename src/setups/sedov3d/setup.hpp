
#pragma once

#include <Kokkos_Core.hpp>
#include <units.hpp>

#include <inih/INIReader.hpp>

#include "default_boundary_setup.hpp"
#include "default_grid_setup.hpp"
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
    double P0;
    double E1;
    double ny;
    double nz;

    explicit ParamSetup(INIReader const& reader)
        : rho0(reader.GetReal("Initialisation", "rho0", 1.0))
        , P0(reader.GetReal("Initialisation", "P0", 1.0))
        , E1(reader.GetReal("Initialisation", "E1", 1.0))
        , ny(reader.GetReal("Grid", "Ny_glob", 1.0))
        , nz(reader.GetReal("Grid", "Nz_glob", 1.0))
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

        auto const& eos = m_eos;
        auto const& param_setup = m_param_setup;
        auto const r = grid.x;
        auto const theta = grid.y;
        auto const phi = grid.z;
        auto const& dv = grid.dv;
        int ny_2 = param_setup.ny / 2;
        int nz_2 = param_setup.nz / 2;


        if (geom == Geometry::Geom_cartesian)
        {
            throw std::runtime_error("No Sedov 3D in Cartesian implemented");
        }

        Kokkos::parallel_for(
            "Sedov_3D_init",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                rho(i, j, k) = param_setup.rho0;
                for (int idim = 0; idim < ndim; ++idim)
                {
                    u(i, j, k, idim) = 0;
                }

                if (r(i) == 1 && j == ny_2  && k == nz_2)
                {
                    double evol = param_setup.E1 / dv(i, j, k);
                    P(i, j, k) = eos.compute_P_from_evol(rho(i, j, k), evol);
                }
                else
                {
                    P(i, j, k) = param_setup.P0;
                }
            });
    }
};

} // namespace novapp
