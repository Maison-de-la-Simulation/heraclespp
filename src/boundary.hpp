//!
//! @file boundary.hpp
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

namespace novapp
{

std::array<std::string, 3> const bc_dir {"_X", "_Y", "_Z"};
std::array<std::string, 2> const bc_face {"_left", "_right"};

class IBoundaryCondition
{
public:
    IBoundaryCondition() = default;

    IBoundaryCondition(int idim, int iface):
    bc_idim(idim),
    bc_iface(iface){};

    IBoundaryCondition(IBoundaryCondition const& x) = default;

    IBoundaryCondition(IBoundaryCondition&& x) noexcept = default;

    virtual ~IBoundaryCondition() noexcept = default;

    IBoundaryCondition& operator=(IBoundaryCondition const& x) = default;

    IBoundaryCondition& operator=(IBoundaryCondition&& x) noexcept = default;

    virtual void execute(KV_double_3d rho,
                          KV_double_4d rhou,
                          KV_double_3d E,
                          KV_double_1d g,
                          Grid const & grid,
                          thermodynamics::PerfectGas const& eos) const = 0;

public:
    int bc_idim;
    int bc_iface;
};

class NullGradient : public IBoundaryCondition
{
    std::string m_label;

public:
    NullGradient(int idim, int iface)
        : IBoundaryCondition(idim, iface)
        , m_label("NullGradient" + bc_dir[idim] + bc_face[iface])
    {
    }
    
    void execute(KV_double_3d rho,
                  KV_double_4d rhou,
                  KV_double_3d E,
                  [[maybe_unused]] KV_double_1d g,
                  Grid const & grid,
                  [[maybe_unused]] thermodynamics::PerfectGas const& eos) const final
    {
        assert(rho.extent(0) == rhou.extent(0));
        assert(rhou.extent(0) == E.extent(0));
        assert(rho.extent(1) == rhou.extent(1));
        assert(rhou.extent(1) == E.extent(1));
        assert(rho.extent(2) == rhou.extent(2));
        assert(rhou.extent(2) == E.extent(2));

        Kokkos::Array<int, 3> begin {0, 0, 0};
        Kokkos::Array<int, 3> end {rho.extent_int(0), rho.extent_int(1), rho.extent_int(2)};

        int const ng = grid.Nghost[bc_idim];
        if (bc_iface == 1)
        {
            begin[bc_idim] = rho.extent_int(bc_idim) - ng;
        }
        end[bc_idim] = begin[bc_idim] + ng;

        int const offset = bc_iface == 0 ? end[bc_idim] : begin[bc_idim] - 1;
        Kokkos::parallel_for(
                m_label,
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
                KOKKOS_CLASS_LAMBDA(int i, int j, int k) 
                {
                    Kokkos::Array<int, 3> offsets {i, j, k};
                    offsets[bc_idim] = offset;
                    rho(i, j, k) = rho(offsets[0], offsets[1], offsets[2]);
                    for (int n = 0; n < rhou.extent_int(3); n++)
                    {
                        rhou(i, j, k, n) = rhou(offsets[0], offsets[1], offsets[2], n);
                    }
                    E(i, j, k) = E(offsets[0], offsets[1], offsets[2]);
                });
    }
};

class PeriodicCondition : public IBoundaryCondition
{
public:

    PeriodicCondition(int idim, int iface)
    : IBoundaryCondition(idim, iface){};    

    void execute([[maybe_unused]]KV_double_3d rho,
                  [[maybe_unused]]KV_double_4d rhou,
                  [[maybe_unused]]KV_double_3d E,
                  [[maybe_unused]] KV_double_1d g,
                  [[maybe_unused]]Grid const & grid,
                  [[maybe_unused]] thermodynamics::PerfectGas const& eos) const final
    {
        // do nothing
    }
};


class ReflexiveCondition : public IBoundaryCondition
{
    std::string m_label;

public:
    ReflexiveCondition(int idim, int iface)
        : IBoundaryCondition(idim, iface)
        , m_label("Reflexive" + bc_dir[idim] + bc_face[iface])
    {
    }

    void execute(KV_double_3d rho,
                  KV_double_4d rhou,
                  KV_double_3d E,
                  [[maybe_unused]] KV_double_1d g,
                  Grid const & grid,
                  [[maybe_unused]] thermodynamics::PerfectGas const& eos) const final
    {
        Kokkos::Array<int, 3> begin {0, 0, 0};
        Kokkos::Array<int, 3> end {rho.extent_int(0), rho.extent_int(1), rho.extent_int(2)};

        int const ng = grid.Nghost[bc_idim];
        if (bc_iface == 1)
        {
            begin[bc_idim] = rho.extent_int(bc_idim) - ng;
        }
        end[bc_idim] = begin[bc_idim] + ng;

        int const mirror = bc_iface == 0 ? (2 * ng - 1) : (2 * (rho.extent(bc_idim) - ng) - 1);
        Kokkos::parallel_for(
        m_label,
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
        KOKKOS_CLASS_LAMBDA(int i, int j, int k) 
        {
            Kokkos::Array<int, 3> offsets {i, j, k};
            offsets[bc_idim] = mirror - offsets[bc_idim];
            rho(i, j, k) = rho(offsets[0], offsets[1], offsets[2]);
            for (int n = 0; n < rhou.extent_int(3); n++)
            {
                rhou(i, j, k, n) = rhou(offsets[0], offsets[1], offsets[2], n);
            }
            rhou(i, j, k, bc_idim) = -rhou(i, j, k, bc_idim);
            E(i, j, k) = E(offsets[0], offsets[1], offsets[2]);
        });
    }
};

class EqHydro : public IBoundaryCondition
{
    std::string m_label;

public:
    EqHydro(int idim, int iface)
        : IBoundaryCondition(idim, iface)
        , m_label("EqHydro" + bc_dir[idim] + bc_face[iface])
    {
    }
    
    void execute(KV_double_3d rho,
                  KV_double_4d rhou,
                  KV_double_3d E,
                  KV_double_1d g,
                  Grid const & grid,
                  thermodynamics::PerfectGas const& eos) const final
    {
        assert(rho.extent(0) == rhou.extent(0));
        assert(rhou.extent(0) == E.extent(0));
        assert(rho.extent(1) == rhou.extent(1));
        assert(rhou.extent(1) == E.extent(1));
        assert(rho.extent(2) == rhou.extent(2));
        assert(rhou.extent(2) == E.extent(2));

        Kokkos::Array<int, 3> begin {0, 0, 0};
        Kokkos::Array<int, 3> end {rho.extent_int(0), rho.extent_int(1), rho.extent_int(2)};

        auto const x_d = grid.x.d_view;
        double mu = eos.compute_mean_molecular_weight();
        double gamma = eos.compute_adiabatic_index();
        double T = eos.compute_const_temprature();
        double kb = units::kb;
        double mh = units::mh;

        double rho0 = 10;

        int const ng = grid.Nghost[bc_idim];
        if (bc_iface == 1)
        {
            begin[bc_idim] = rho.extent_int(bc_idim) - ng;
        }
        end[bc_idim] = begin[bc_idim] + ng;

        Kokkos::parallel_for(
        m_label,
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(begin, end),
        KOKKOS_CLASS_LAMBDA(int i, int j, int k) 
        {
            double xcenter = x_d(i) + grid.dx[0] / 2;
            double x0 = kb * T / (mu * mh * std::abs(g(0)));
            rho(i, j, k) = rho0 * Kokkos::exp(- xcenter / x0);
            for (int n = 0; n < rhou.extent_int(3); n++)
            {
                rhou(i, j, k, n) = 0;
            }
            E(i, j, k) = rho(i, j, k) * kb * T / (mu * mh * (gamma - 1));
        });
    }
};

inline std::unique_ptr<IBoundaryCondition> factory_boundary_construction(
    std::string const& s,
    int idim, int iface)
{
    if (s == "NullGradient")
    {
        return std::make_unique<NullGradient>(idim, iface);
    }
    if (s == "Periodic")
    {
        return std::make_unique<PeriodicCondition>(idim, iface);
    }
    if (s == "Reflexive")
    {
        return std::make_unique<ReflexiveCondition>(idim, iface);
    }
    if (s == "EqHydro")
    {
        return std::make_unique<EqHydro>(idim, iface);
    }
    throw std::runtime_error("Unknown boundary condition : " + s + ".");
}


class DistributedBoundaryCondition
{
private:
    std::array<KDV_double_4d, ndim> m_mpi_buffer;

    std::array<std::unique_ptr<IBoundaryCondition>, ndim * 2> m_bcs;

    std::array<int, ndim*2> m_bc_order;

    void ghostFill(
            KV_double_3d rho,
            KV_double_4d rhou,
            KV_double_3d E,
            int bc_idim,
            int bc_iface,
            Grid const& grid) const;

    void generate_order(INIReader const& reader)
    {
        std::string bc_priority = reader.Get("Boundary Condition", "priority", "");

        if(bc_priority.empty())
        {
            std::iota(m_bc_order.begin(), m_bc_order.end(), 0); // m_bc_order = {0,1,2,3,4,5};
            return;
        }
        std::array<std::string, ndim*2> tmp_arr;
        std::stringstream ssin(bc_priority);
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
            Grid const& grid)
    {
        std::string bc_choice;
        std::string bc_choice_dir;
        std::array<std::string, ndim*2> bc_choice_faces;

        bc_choice = reader.Get("Boundary Condition", "BC", "");
        for(int idim = 0; idim < ndim; idim++)
        {
            bc_choice_dir = reader.Get("Boundary Condition", "BC"+bc_dir[idim], bc_choice);
            for (int iface = 0; iface < 2; iface++)
            {
                bc_choice_faces[idim*2+iface] = reader.Get("Boundary Condition", "BC"+bc_dir[idim]+bc_face[iface], bc_choice_dir);
                if(bc_choice_faces[idim*2+iface].empty() )
                {
                    throw std::runtime_error("boundary condition not fully defined for dimension "+bc_dir[idim]);
                }
            }
        }

        for(int idim=0; idim<ndim; idim++)
        {
            for(int iface=0; iface<2; iface++)
            {
                if (!grid.is_border[idim][iface])
                {
                    m_bcs[idim * 2 + iface] = factory_boundary_construction("Periodic", idim, iface);
                }
                else
                {
                    m_bcs[idim*2+iface] = factory_boundary_construction(bc_choice_faces[idim*2+iface], idim, iface);
                }
            }
        }

        std::array<int, 3> buf_size = grid.Nx_local_wg;
        for (int idim = 0; idim < ndim; idim++)
        {
            buf_size[idim] = grid.Nghost[idim];
            m_mpi_buffer[idim] = KDV_double_4d("", buf_size[0], buf_size[1], buf_size[2], ndim + 2);
            buf_size[idim] = grid.Nx_local_wg[idim];
        }

        generate_order(reader);
    }

    void execute(KV_double_3d rho, KV_double_4d rhou, KV_double_3d E, KV_double_1d g, Grid const & grid, thermodynamics::PerfectGas const& eos) const
    {
        for (int idim = 0; idim < ndim; idim++)
        {
            for (int iface = 0; iface < 2; iface++)
            {
                ghostFill(rho, rhou, E, idim, iface, grid);
            }
        }

        for ( int const bc_id : m_bc_order )
        {
            m_bcs[bc_id]->execute(rho, rhou, E, g, grid, eos);
        }
    }
};

} // namespace novapp
