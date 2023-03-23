//!
//! @file boundary.hpp
//!

#pragma once

#include "Kokkos_shortcut.hpp"
#include "buffer.hpp"
#include "grid.hpp"
#include "ndim.hpp"

namespace novapp
{

class IBoundaryCondition
{
public:
    IBoundaryCondition() = default;

    IBoundaryCondition(Grid const & grid, int idim, int iface):
    sbuf(grid.Nghost, grid.Nx_local_ng, 2+ndim),
    rbuf(grid.Nghost, grid.Nx_local_ng, 2+ndim),
    bc_idim(idim),
    bc_iface(iface)
    {};

    IBoundaryCondition(IBoundaryCondition const& x) = default;

    IBoundaryCondition(IBoundaryCondition&& x) noexcept = default;

    virtual ~IBoundaryCondition() noexcept = default;

    IBoundaryCondition& operator=(IBoundaryCondition const& x) = default;

    IBoundaryCondition& operator=(IBoundaryCondition&& x) noexcept = default;

    virtual void execute(KV_double_3d rho,
                          KV_double_4d rhou,
                          KV_double_3d E,
                          Grid const & grid) const = 0;

    void ghostFill(KV_double_3d rho,
                       KV_double_4d rhou,
                       KV_double_3d E, 
                       Grid const & grid);

private:    
    Buffer sbuf, rbuf;
public:
    int bc_idim;
    int bc_iface;

};


class NullGradient : public IBoundaryCondition
{
    std::string m_label;

public:
    NullGradient(Grid const & grid, int idim, int iface)
        : IBoundaryCondition(grid, idim, iface)
        , m_label("NullGradient_" + std::to_string(idim) + "_" + std::to_string(iface))
    {
    }
    
    void execute(KV_double_3d rho,
                  KV_double_4d rhou,
                  KV_double_3d E,
                  Grid const & grid) const final
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
                KOKKOS_LAMBDA(int i, int j, int k) {
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

    PeriodicCondition(Grid const & grid, int idim, int iface)
    : IBoundaryCondition(grid, idim, iface){};    

    void execute([[maybe_unused]]KV_double_3d rho,
                  [[maybe_unused]]KV_double_4d rhou,
                  [[maybe_unused]]KV_double_3d E,
                  [[maybe_unused]]Grid const & grid) const final
    {
        // ghostFill_dev(rho, rhou, E, grid);
        // do nothing
    }

};


class ReflexiveCondition : public IBoundaryCondition
{
    std::string m_label;

public:
    ReflexiveCondition(Grid const & grid, int idim, int iface)
        : IBoundaryCondition(grid, idim, iface)
        , m_label("Reflexive_" + std::to_string(idim) + "_" + std::to_string(iface))
    {
    }

    void execute(KV_double_3d rho,
                  KV_double_4d rhou,
                  KV_double_3d E,
                  Grid const & grid) const final
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
                KOKKOS_LAMBDA(int i, int j, int k) {
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

inline std::unique_ptr<IBoundaryCondition> factory_boundary_construction(
    Grid const & grid,
    std::string const& s,
    int idim, int iface)
{
    if (s == "NullGradient")
    {
        return std::make_unique<NullGradient>(grid, idim, iface);
    }
    if (s == "Periodic")
    {
        return std::make_unique<PeriodicCondition>(grid, idim, iface);
    }
    if (s == "Reflexive")
    {
        return std::make_unique<ReflexiveCondition>(grid, idim, iface);
    }
    throw std::runtime_error("Unknown boundary condition : " + s + ".");
}

} // namespace novapp
