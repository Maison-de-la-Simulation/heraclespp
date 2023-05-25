//!
//! @file boundary_factory.hpp
//!

#pragma once

#include <algorithm>
#include <iostream>
#include <numeric>
#include <sstream>

#include <PerfectGas.hpp>

#include "Kokkos_shortcut.hpp"
#include "grid.hpp"
#include "ndim.hpp"
#include "units.hpp"
#include "nova_params.hpp"
#include "boundary.hpp"
#include "setup.hpp"
#include "nova_params.hpp"

namespace novapp
{

inline std::unique_ptr<IBoundaryCondition> factory_boundary_construction(
    std::string const& boundary,
    int idim,
    int iface,
    thermodynamics::PerfectGas const& eos,
    Grid const& grid,
    ParamSetup const& param_setup)
{
    if (boundary == "NullGradient")
    {
        return std::make_unique<NullGradient>(idim, iface, grid);
    }
    if (boundary == "Periodic")
    {
        return std::make_unique<PeriodicCondition>(idim, iface);
    }
    if (boundary == "Reflexive")
    {
        return std::make_unique<ReflexiveCondition>(idim, iface, grid);
    }
    if (boundary == "UserDefined")
    {
        return std::make_unique<BoundarySetup>(idim, iface, eos, grid, param_setup);
    }
    throw std::runtime_error("Unknown boundary condition : " + boundary + ".");
}

class DistributedBoundaryCondition
{
private:
    std::array<KDV_double_4d, ndim> m_mpi_buffer;
    std::array<std::unique_ptr<IBoundaryCondition>, ndim * 2> m_bcs;
    std::array<int, ndim*2> m_bc_order;
    thermodynamics::PerfectGas m_eos;
    Grid m_grid;
    Param m_param;
    ParamSetup m_param_setup;

    void ghostFill(
            KV_double_3d rho,
            KV_double_4d rhou,
            KV_double_3d E,
            KV_double_4d fx,
            int bc_idim,
            int bc_iface) const;

    void generate_order()
    {
        if(m_param.bc_priority.empty())
        {
            std::iota(m_bc_order.begin(), m_bc_order.end(), 0); // m_bc_order = {0,1,2,3,4,5};
            return;
        }
        std::array<std::string, ndim*2> tmp_arr;
        std::stringstream ssin(m_param.bc_priority);
        for(int i=0; i<ndim*2 && ssin.good(); i++)
        {
            ssin >> tmp_arr[i];
        }

        int counter=0;
        for(int i=0; i<ndim*2; i++)
        {
            if(tmp_arr[i] == "X_left") {m_bc_order[0] = i; counter++;}
            else if(tmp_arr[i] == "X_right") {m_bc_order[1] = i; counter++;}
            else if(tmp_arr[i] == "Y_left" && ndim>=2) {m_bc_order[2] = i; counter++;}
            else if(tmp_arr[i] == "Y_right" && ndim>=2) {m_bc_order[3] = i; counter++;}
            else if(tmp_arr[i] == "Z_left" && ndim==3) {m_bc_order[4] = i; counter++;}
            else if(tmp_arr[i] == "Z_right" && ndim==3) {m_bc_order[5] = i; counter++;}
        }
        std::reverse(m_bc_order.begin(), m_bc_order.end());
        if(counter!=ndim*2)
        {
            throw std::runtime_error("boundary priority not fully defined !");
        }
    }

public:
    DistributedBoundaryCondition(
            INIReader const& reader,
            thermodynamics::PerfectGas const& eos,
            Grid const& grid, 
            Param const& param,
            ParamSetup const& param_setup)
            :  m_eos(eos)
            , m_grid(grid)
            , m_param(param)
            , m_param_setup(param_setup)
    {
        std::string bc_choice_dir;
        std::array<std::string, ndim*2> bc_choice_faces;

        for(int idim = 0; idim < ndim; idim++)
        {
            bc_choice_dir = reader.Get("Boundary Condition", "BC"+bc_dir[idim], m_param.bc_choice);
            for (int iface = 0; iface < 2; iface++)
            {
                bc_choice_faces[idim*2+iface] = reader.Get("Boundary Condition",
                                                           "BC"+bc_dir[idim]+bc_face[iface],
                                                           bc_choice_dir);
                if(bc_choice_faces[idim*2+iface].empty() )
                {
                    throw std::runtime_error("boundary condition not fully defined for dimension "
                                              +bc_dir[idim]);
                }
            }
        }

        for(int idim=0; idim<ndim; idim++)
        {
            for(int iface=0; iface<2; iface++)
            {
                if (!m_grid.is_border[idim][iface])
                {
                    m_bcs[idim * 2 + iface] = factory_boundary_construction(
                                               "Periodic", idim, iface, m_eos, m_grid, m_param_setup);
                }
                else
                {
                    m_bcs[idim*2+iface] = factory_boundary_construction(
                                           bc_choice_faces[idim*2+iface], idim, iface, m_eos, m_grid, m_param_setup);
                }
            }
        }
        
        std::array<int, 3> buf_size = grid.Nx_local_wg;
        for (int idim = 0; idim < ndim; idim++)
        {
            buf_size[idim] = grid.Nghost[idim];
            m_mpi_buffer[idim] = KDV_double_4d("", buf_size[0], buf_size[1], buf_size[2], ndim + 2 + param.nfx);
            buf_size[idim] = grid.Nx_local_wg[idim];
        }

        generate_order();
    }

    void execute(KV_double_3d rho,
                 KV_double_4d rhou,
                 KV_double_3d E,
                 KV_double_4d fx,
                 KV_double_1d g) const
    {
        for (int idim = 0; idim < ndim; idim++)
        {
            for (int iface = 0; iface < 2; iface++)
            {
                ghostFill(rho, rhou, E, fx, idim, iface);
            }
        }

        for ( int const bc_id : m_bc_order )
        {
            m_bcs[bc_id]->execute(rho, rhou, E, fx, g);
        }
    }
};

} // namespace novapp
