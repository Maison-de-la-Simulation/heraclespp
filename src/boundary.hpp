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

class ParamSetup;

class IBoundaryCondition
{
public:
    IBoundaryCondition() = default;

    IBoundaryCondition(int idim, int iface)
        : bc_idim(idim)
        , bc_iface(iface){};

    IBoundaryCondition(IBoundaryCondition const& x) = default;

    IBoundaryCondition(IBoundaryCondition&& x) noexcept = default;

    virtual ~IBoundaryCondition() noexcept = default;

    IBoundaryCondition& operator=(IBoundaryCondition const& x) = default;

    IBoundaryCondition& operator=(IBoundaryCondition&& x) noexcept = default;

    virtual void execute([[maybe_unused]] KV_double_3d rho,
                         [[maybe_unused]] KV_double_4d rhou,
                         [[maybe_unused]] KV_double_3d E,
                         [[maybe_unused]] KV_double_4d fx,
                         [[maybe_unused]] KV_double_1d g) const 
                         {
                            throw std::runtime_error("Boundary not implemented");
                         }

public:
    int bc_idim;
    int bc_iface;
};

class NullGradient : public IBoundaryCondition
{
    std::string m_label;

    Grid m_grid;

public:
    NullGradient(int idim, int iface,
        Grid const& grid)
        : IBoundaryCondition(idim, iface)
        , m_label("NullGradient" + bc_dir[idim] + bc_face[iface])
        , m_grid(grid)
    {
    }
    
    void execute(KV_double_3d rho,
                 KV_double_4d rhou,
                 KV_double_3d E,
                 [[maybe_unused]] KV_double_4d fx,
                 [[maybe_unused]] KV_double_1d g) const final
    {
        assert(rho.extent(0) == rhou.extent(0));
        assert(rhou.extent(0) == E.extent(0));
        assert(rho.extent(1) == rhou.extent(1));
        assert(rhou.extent(1) == E.extent(1));
        assert(rho.extent(2) == rhou.extent(2));
        assert(rhou.extent(2) == E.extent(2));

        Kokkos::Array<int, 3> begin {0, 0, 0};
        Kokkos::Array<int, 3> end {rho.extent_int(0), rho.extent_int(1), rho.extent_int(2)};
        int nfx = fx.extent_int(3);

        int const ng = m_grid.Nghost[bc_idim];
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
            for (int ifx = 0; ifx < nfx; ++ifx)
            {
                fx(i, j, k, ifx) = fx(offsets[0], offsets[1], offsets[2], ifx);
            }
        });
    }
};

class PeriodicCondition : public IBoundaryCondition
{
public:
    PeriodicCondition(int idim, int iface)
        : IBoundaryCondition(idim, iface){}

    void execute([[maybe_unused]] KV_double_3d rho,
                 [[maybe_unused]] KV_double_4d rhou,
                 [[maybe_unused]] KV_double_3d E,
                 [[maybe_unused]] KV_double_4d fx,
                 [[maybe_unused]] KV_double_1d g) const final
    {
        // do nothing
    }
};


class ReflexiveCondition : public IBoundaryCondition
{
    std::string m_label;

private:
    Grid m_grid;

public:
    ReflexiveCondition(int idim, int iface,
        Grid const& grid)
        : IBoundaryCondition(idim, iface)
        , m_label("Reflexive" + bc_dir[idim] + bc_face[iface])
        , m_grid(grid)
        
    {
    }

    void execute(KV_double_3d rho,
                 KV_double_4d rhou,
                 KV_double_3d E,
                 [[maybe_unused]] KV_double_4d fx,
                 [[maybe_unused]] KV_double_1d g) const final
    {
        Kokkos::Array<int, 3> begin {0, 0, 0};
        Kokkos::Array<int, 3> end {rho.extent_int(0), rho.extent_int(1), rho.extent_int(2)};
        int nfx = fx.extent_int(3);

        int const ng = m_grid.Nghost[bc_idim];
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
            for (int ifx = 0; ifx < nfx; ++ifx)
            {
                fx(i, j, k, ifx) = fx(offsets[0], offsets[1], offsets[2], ifx);
            }
        });
    }
};

} // namespace novapp
