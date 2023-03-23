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
public:
    NullGradient(Grid const & grid, int idim, int iface)
    : IBoundaryCondition(grid, idim, iface){};
    
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

        if(bc_idim == 0)
        {
            int ng = grid.Nghost[0];
            if(bc_iface==0)
            {
                Kokkos::parallel_for("NullGradient_X_left",
                                     Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                                     {0, 0, 0},
                                     {ng, rho.extent_int(1), rho.extent_int(2)}),
                                     KOKKOS_LAMBDA(int i, int j, int k)
                {
                    rho(i, j, k) = rho(ng, j, k);
                    for (int n=0; n<rhou.extent_int(3); n++)
                    {
                        rhou(i, j, k, n) = rhou(ng, j, k, n);
                    }
                    E(i, j, k) = E(ng, j, k);
                });
            }
            else
            {
                int offset = rho.extent_int(0)-ng-1;
                Kokkos::parallel_for("NullGradient_X_right",
                                     Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                                     {rho.extent_int(0)-ng, 0, 0},
                                     {rho.extent_int(0), rho.extent_int(1), rho.extent_int(2)}),
                                     KOKKOS_LAMBDA(int i, int j, int k)
                {
                    rho(i, j, k) = rho(offset, j, k);
                    for (int n=0; n<rhou.extent_int(3); n++)
                    {
                        rhou(i, j, k, n) = rhou(offset, j, k, n);
                    }
                    E(i, j, k) = E(offset, j, k);
                });
            }
        }
        else if(bc_idim == 1)
        {
            int ng = grid.Nghost[1];
            if(bc_iface==0)
            {
                Kokkos::parallel_for("NullGradient_Y_left",
                                     Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                                     {0, 0, 0},
                                     {rho.extent_int(0), ng, rho.extent_int(2)}),
                                     KOKKOS_LAMBDA(int i, int j, int k)
                {
                    rho(i, j, k) = rho(i, ng, k);
                    for (int n=0; n<rhou.extent_int(3); n++)
                    {
                        rhou(i, j, k, n) = rhou(i, ng, k, n);
                    }
                    E(i, j, k) = E(i, ng, k);
                });
            }
            else
            {
                int offset = rho.extent_int(1)-ng-1;
                Kokkos::parallel_for("NullGradient_Y_right",
                                     Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                                     {0, rho.extent_int(1)-ng, 0},
                                     {rho.extent_int(0), rho.extent_int(1), rho.extent_int(2)}),
                                     KOKKOS_LAMBDA(int i, int j, int k)
                {
                    rho(i, j, k) = rho(i, offset, k);
                    for (int n=0; n<rhou.extent_int(3); n++)
                    {
                        rhou(i, j, k, n) = rhou(i, offset, k, n);
                    }
                    E(i, j, k) = E(i, offset, k);
                });
            }
        }
        else // bc_idim == 2
        {
            int ng = grid.Nghost[2];
            if(bc_iface==0)
            {
                Kokkos::parallel_for("NullGradient_Z_left",
                                     Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                                     {0, 0, 0},
                                     {rho.extent_int(0), rho.extent_int(1), ng}),
                                     KOKKOS_LAMBDA(int i, int j, int k)
                {
                    rho(i, j, k) = rho(i, j, ng);
                    for (int n=0; n<rhou.extent_int(3); n++)
                    {
                        rhou(i, j, k, n) = rhou(i, j, ng, n);
                    }
                    E(i, j, k) = E(i, j, ng);
                });
            }
            else
            {
                int offset = rho.extent_int(2)-ng-1;
                Kokkos::parallel_for("NullGradient_Z_right",
                                     Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                                     {0, 0, rho.extent_int(2)-ng},
                                     {rho.extent_int(0), rho.extent_int(1), rho.extent_int(2)}),
                                     KOKKOS_LAMBDA(int i, int j, int k)
                {
                    rho(i, j, k) = rho(i, j, offset);
                    for (int n=0; n<rhou.extent_int(3); n++)
                    {
                        rhou(i, j, k, n) = rhou(i, j, offset, n);
                    }
                    E(i, j, k) = E(i, j, offset);
                });
            }
        }
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
public:
    ReflexiveCondition(Grid const & grid, int idim, int iface)
    : IBoundaryCondition(grid, idim, iface){};

    void execute(KV_double_3d rho,
                  KV_double_4d rhou,
                  KV_double_3d E,
                  Grid const & grid) const final
    {
        if(bc_idim==0)
        {
            int ng = grid.Nghost[0];
            if(bc_iface==0)
            {
                Kokkos::parallel_for("Reflexive_X_left",
                                 Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                                 {0, 0, 0},
                                 {ng, rho.extent_int(1), rho.extent_int(2)}),
                                 KOKKOS_LAMBDA(int i, int j, int k)
                {
                    rho(i, j, k) = rho(2*ng-1-i, j, k);
                    for (int n=0; n<rhou.extent_int(3); n++)
                    {    
                        rhou(i, j, k, n) = (n==0? -rhou(2*ng-1-i, j, k, n):rhou(2*ng-1-i, j, k, n));
                    }

                    E(i, j, k) = E(2*ng-1-i, j, k);
                });
            }
            else
            {
                int offset=2*(rho.extent(0)-ng)-1;
                Kokkos::parallel_for("Reflexive_X_right",
                                     Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                                     {rho.extent_int(0)-ng, 0, 0},
                                     {rho.extent_int(0), rho.extent_int(1), rho.extent_int(2)}),
                                     KOKKOS_LAMBDA(int i, int j, int k)
                {
                    rho(i, j, k) = rho(offset-i, j, k);
                    for (int n=0; n<rhou.extent_int(3); n++)
                    {    
                        rhou(i, j, k, n) = (n==0? -rhou(offset-i, j, k, n):rhou(offset-i, j, k, n));
                    }
                    E(i, j, k) = E(offset-i, j, k);
                });
            }
        }
        else if(bc_idim==1)
        {
            int ng = grid.Nghost[1];
            if(bc_iface==0)
            {
                Kokkos::parallel_for("Reflexive_Y_left",
                                     Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                                     {0, 0, 0},
                                     {rho.extent_int(0), ng, rho.extent_int(2)}),
                                     KOKKOS_LAMBDA(int i, int j, int k)
                {
                    rho(i, j, k) = rho(i, 2*ng-1-j, k);
                    for (int n=0; n<rhou.extent_int(3); n++)
                    {    
                        rhou(i, j, k, n) = (n==1? -rhou(i, 2*ng-1-j, k, n):rhou(i, 2*ng-1-j, k, n));
                    }
                    E(i, j, k) = E(i, 2*ng-1-j, k);
                });
            }
            else
            {
                int offset=2*(rho.extent(1)-ng)-1;
                Kokkos::parallel_for("Reflexive_Y_right",
                                     Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                                     {0, rho.extent_int(1)-ng, 0},
                                     {rho.extent_int(0), rho.extent_int(1), rho.extent_int(2)}),
                                     KOKKOS_LAMBDA(int i, int j, int k)
                {
                    rho(i, j, k) = rho(i, offset-j, k);
                    for (int n=0; n<rhou.extent_int(3); n++)
                    {    
                        rhou(i, j, k, n) = (n==1? -rhou(i, offset-j, k, n):rhou(i, offset-j, k, n));
                    }
                    E(i, j, k) = E(i, offset-j, k);
                });
            }
        }
        else
        {
            int ng = grid.Nghost[2];
            if(bc_iface==0)
            {
                Kokkos::parallel_for("Reflexive_Z_left",
                                     Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                                     {0, 0, 0},
                                     {rho.extent_int(0), rho.extent_int(1), ng}),
                                     KOKKOS_LAMBDA(int i, int j, int k)
                {
                    rho(i, j, k) = rho(i, j, 2*ng-1-k);
                    for (int n=0; n<rhou.extent_int(3); n++)
                    {    
                        rhou(i, j, k, n) = (n==2? -rhou(i, j, 2*ng-1-k, n):rhou(i, j, 2*ng-1-k, n));
                    }
                    rhou(i, j, k, 0) = rhou(i, j, 2*ng-1-k, 0);
                    rhou(i, j, k, 1) = rhou(i, j, 2*ng-1-k, 1);
                    rhou(i, j, k, 2) = -rhou(i, j, 2*ng-1-k, 2);
                    E(i, j, k) = E(i, j, 2*ng-1-k);
                });
            }
            else
            {
                int offset=2*(rho.extent(2)-ng)-1;
                Kokkos::parallel_for("Reflexive_Z_right",
                                     Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                                     {0, 0, rho.extent_int(2)-ng},
                                     {rho.extent_int(0), rho.extent_int(1), rho.extent_int(2)}),
                                     KOKKOS_LAMBDA(int i, int j, int k)
                {
                    rho(i, j, k) = rho(i, j, offset-k);
                    for (int n=0; n<rhou.extent_int(3); n++)
                    {    
                        rhou(i, j, k, n) = (n==2? -rhou(i, j, offset-k, n):rhou(i, j, offset-k, n));
                    }
                    E(i, j, k) = E(i, j, offset-k);
                });
            }
        }
    }
};

class xPeriodicyReflexiveCondition : public IBoundaryCondition
{
public:
    xPeriodicyReflexiveCondition(Grid const & grid, int idim, int iface)
    : IBoundaryCondition(grid, idim, iface){};
    
    void execute([[maybe_unused]]KV_double_3d rho,
                 [[maybe_unused]]KV_double_4d rhou,
                 [[maybe_unused]]KV_double_3d E,
                 [[maybe_unused]]Grid const & grid) const final
    {
        // not implemented yet
    }
};

class xLeftReflexivexRightNullGradientCondition : public IBoundaryCondition
{
public:
    xLeftReflexivexRightNullGradientCondition(Grid const & grid) : IBoundaryCondition(grid){};
    
    void bcUpdate(KV_double_3d rho,
                  KV_double_4d rhou,
                  KV_double_3d E,
                  Grid const & grid) const final
    {
        int ng=0;
        int size_x=0;
        if(grid.is_border[0][0])
        {
            ng = grid.Nghost[0];
            Kokkos::parallel_for("Reflexive_implementation",
                                 Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                                 {0, 0, 0},
                                 {ng, rho.extent_int(1), rho.extent_int(2)}),
                                 KOKKOS_LAMBDA(int i, int j, int k)
            {
                rho(i, j, k) = rho(2*ng-1-i, j, k);
                for (int n=0; n<rhou.extent_int(3); n++)
                {    
                    rhou(i, j, k, n) = (n==grid.Ndim-1? -rhou(2*ng-1-i, j, k, n):rhou(2*ng-1-i, j, k, n));
                }
                E(i, j, k) = E(2*ng-1-i, j, k);
            });
        }
        if(grid.is_border[0][1])
        {
            ng = grid.Nghost[0];
            size_x = rho.extent(0)-1;
            Kokkos::parallel_for("NullGradient_implementation",
                                 Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                                 {0, 0, 0},
                                 {ng, rho.extent_int(1), rho.extent_int(2)}),
                                 KOKKOS_LAMBDA(int i, int j, int k)
            {
                rho(size_x-i, j, k) = rho(size_x-ng, j, k);
                for (int n=0; n<rhou.extent_int(3); n++)
                {
                    rhou(size_x-i, j, k, n) = rhou(size_x-ng, j, k, n);
                }
                E(size_x-i, j, k) = E(size_x-ng, j, k);
            });
        }
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
    if (s == "xLeftReflexivexRightNullGradient")
    {
        return std::make_unique<xLeftReflexivexRightNullGradientCondition>(grid);
    }
    throw std::runtime_error("Unknown boundary condition : " + s + ".");
}

} // namespace novapp
