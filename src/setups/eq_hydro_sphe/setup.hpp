
#pragma once

#include <Kokkos_Core.hpp>
#include <units.hpp>

#include <inih/INIReader.hpp>

#include "default_grid_setup.hpp"
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
    double T;
    double M;

    explicit ParamSetup(INIReader const& reader)
        : rho0(reader.GetReal("Initialisation", "rho0", 1.0))
        , u0(reader.GetReal("Initialisation", "u0", 1.0))
        , T(reader.GetReal("Initialisation", "T", 100.))
        , M(reader.GetReal("Gravity", "M", 1.0))
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
        ParamSetup const& param_setup,
        Gravity const& gravity)
        : m_eos(eos)
        , m_grid(grid)
        , m_param_setup(param_setup)
    {
    }

    void execute(
        Range const& range,
        KV_double_3d const rho,
        KV_double_4d const u,
        KV_double_3d const P,
        [[maybe_unused]] KV_double_4d fx) const final
    {
        assert(rho.extent(0) == u.extent(0));
        assert(u.extent(0) == P.extent(0));
        assert(rho.extent(1) == u.extent(1));
        assert(u.extent(1) == P.extent(1));
        assert(rho.extent(2) == u.extent(2));
        assert(u.extent(2) == P.extent(2));

        auto const xc = m_grid.x_center;
        double const mu = m_eos.mean_molecular_weight();

        std::cout <<"Scale = " << units::kb * m_param_setup.T
                / (mu * units::mp * units::G * m_param_setup.M)<< std::endl;
        
        auto const [begin, end] = cell_range(range);
        Kokkos::parallel_for(
        "eq_hydro_init",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
        KOKKOS_CLASS_LAMBDA(int i, int j, int k)
        {
            double x0 = units::kb * m_param_setup.T / (mu * units::mp * units::G * m_param_setup.M);

            rho(i, j, k) = m_param_setup.rho0 * Kokkos::exp(1. / (xc(i) * x0));

            for (int idim = 0; idim < ndim; ++idim)
            {
                u(i, j, k, idim) = m_param_setup.u0;
            }

            P(i, j, k) = m_eos.compute_P_from_T(rho(i, j, k), m_param_setup.T);
        });
    }
};

template <class Gravity>
class BoundarySetup : public IBoundaryCondition
{
    std::string m_label;

private:
    EOS m_eos;
    Grid m_grid;
    ParamSetup m_param_setup;

public:
    BoundarySetup(int idim, int iface,
        EOS const& eos,
        Grid const& grid,
        ParamSetup const& param_setup,
        Gravity const& gravity)
        : IBoundaryCondition(idim, iface)
        , m_label("UserDefined" + bc_dir[idim] + bc_face[iface])
        , m_eos(eos)
        , m_grid(grid)
        , m_param_setup(param_setup)
    {
    }
    
    void execute(KV_double_3d rho,
                 KV_double_4d rhou,
                 KV_double_3d E,
                 [[maybe_unused]] KV_double_4d fx) const final
    {
        assert(rho.extent(0) == rhou.extent(0));
        assert(rhou.extent(0) == E.extent(0));
        assert(rho.extent(1) == rhou.extent(1));
        assert(rhou.extent(1) == E.extent(1));
        assert(rho.extent(2) == rhou.extent(2));
        assert(rhou.extent(2) == E.extent(2));

        Kokkos::Array<int, 3> begin {0, 0, 0};
        Kokkos::Array<int, 3> end {rho.extent_int(0), rho.extent_int(1), rho.extent_int(2)};

        auto const xc = m_grid.x_center;
        double const mu = m_eos.mean_molecular_weight();

        int const ng = m_grid.Nghost[m_bc_idim];
        if (m_bc_iface == 1)
        {
            begin[m_bc_idim] = rho.extent_int(m_bc_idim) - ng;
        }
        end[m_bc_idim] = begin[m_bc_idim] + ng;

        Kokkos::parallel_for(
        m_label,
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
        KOKKOS_CLASS_LAMBDA(int i, int j, int k) 
        {
            double x0 = units::kb * m_param_setup.T / (mu * units::mp * units::G * m_param_setup.M);
            rho(i, j, k) = m_param_setup.rho0 * Kokkos::exp(1. / (xc(i) * x0));
            for (int n = 0; n < rhou.extent_int(3); n++)
            {
                rhou(i, j, k, n) = m_param_setup.rho0 * m_param_setup.u0;
            }
            E(i, j, k) = m_eos.compute_evol_from_T(rho(i, j, k), m_param_setup.T);
        });
    }
};

} // namespace novapp
