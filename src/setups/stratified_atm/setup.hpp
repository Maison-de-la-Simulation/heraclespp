
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

    explicit ParamSetup(INIReader const& reader)
        : rho0(reader.GetReal("Initialisation", "rho0", 1.0))
        , u0(reader.GetReal("Initialisation", "u0", 1.0))
        , T(reader.GetReal("Perfect Gas", "temperature", 100.))
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
        ParamSetup const& param_setup,
        Gravity const& gravity)
        : m_eos(eos)
        , m_grid(grid)
        , m_param_setup(param_setup)
        , m_gravity(gravity)
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
        auto const& eos = m_eos;
        auto const& gravity = m_gravity;
        auto const& param_setup = m_param_setup;
        double mu = m_eos.mean_molecular_weight();
        /* std::cout <<"Scale = " << units::kb * m_param_setup.T
            / (mu * units::mp * Kokkos::fabs(g(0))) << std::endl; */

        Kokkos::parallel_for(
            "stratified_atm_init",
            cell_mdrange(range),
            KOKKOS_LAMBDA(int i, int j, int k)
            {
                double x0 = units::kb * param_setup.T * units::Kelvin
                        / (mu * units::mp * Kokkos::fabs(gravity(i, j, k, 0)) * units::acc);

                rho(i, j, k) = param_setup.rho0 * units::density * Kokkos::exp(- xc(i) / x0);

                for (int idim = 0; idim < ndim; ++idim)
                {
                    u(i, j, k, idim) = param_setup.u0 * units::velocity;
                }

                P(i, j, k) = eos.compute_P_from_T(rho(i, j, k), param_setup.T);
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
    Gravity m_gravity;

public:
    BoundarySetup(int idim, int iface,
        [[maybe_unused]] EOS const& eos,
        [[maybe_unused]] Grid const& grid,
        [[maybe_unused]] ParamSetup const& param_setup,
        [[maybe_unused]] Gravity const& gravity)
        : IBoundaryCondition(idim, iface)
        , m_label("UserDefined" + bc_dir[idim] + bc_face[iface])
        , m_eos(eos)
        , m_grid(grid)
        , m_param_setup(param_setup)
        , m_gravity(gravity)
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
        auto const& eos = m_eos;
        auto const& gravity = m_gravity;
        auto const& param_setup = m_param_setup;
        double mu = m_eos.mean_molecular_weight();

        int const ng = m_grid.Nghost[m_bc_idim];
        if (m_bc_iface == 1)
        {
            begin[m_bc_idim] = rho.extent_int(m_bc_idim) - ng;
        }
        end[m_bc_idim] = begin[m_bc_idim] + ng;

        Kokkos::parallel_for(
            m_label,
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
            KOKKOS_LAMBDA(int i, int j, int k) 
            {
                double gravity_x = gravity(i, j, k, 0) * units::acc;

                double x0 = units::kb * param_setup.T * units::Kelvin
                        / (mu * units::mp * Kokkos::fabs(gravity_x));

                rho(i, j, k) = param_setup.rho0 * units::density * Kokkos::exp(- xc(i) / x0);

                for (int n = 0; n < rhou.extent_int(3); n++)
                {
                    rhou(i, j, k, n) = param_setup.rho0 * units::density * param_setup.u0 * units::velocity;
                }
                
                E(i, j, k) = eos.compute_evol_from_T(rho(i, j, k) * units::density, param_setup.T * units::Kelvin);
            });
    }
};

} // namespace novapp
