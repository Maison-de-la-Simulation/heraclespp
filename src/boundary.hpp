//!
//! @file boundary.hpp
//!
#pragma once

#include <Kokkos_Core.hpp>
#include "buffer.hpp"
#include "grid.hpp"
#include "ndim.hpp"

class IBoundaryCondition
{
public:
    IBoundaryCondition() = default;

    IBoundaryCondition(Grid const & grid):
    sbuf(grid.Nghost, grid.Nx_local_ng, 2+ndim),
    rbuf(grid.Nghost, grid.Nx_local_ng, 2+ndim)
    {};

    IBoundaryCondition(IBoundaryCondition const& x) = default;

    IBoundaryCondition(IBoundaryCondition&& x) noexcept = default;

    virtual ~IBoundaryCondition() noexcept = default;

    IBoundaryCondition& operator=(IBoundaryCondition const& x) = default;

    IBoundaryCondition& operator=(IBoundaryCondition&& x) noexcept = default;

    virtual void bcUpdate(Kokkos::View<double***> rho,
                               Kokkos::View<double****> rhou,
                               Kokkos::View<double***> E,
                               Grid const & grid) const = 0;

    void ghostExchange(Kokkos::View<double***> rho,
                      Kokkos::View<double****> rhou,
                      Kokkos::View<double***> E, 
                      Grid const & grid);

    void execute(Kokkos::View<double***> rho,
                               Kokkos::View<double****> rhou,
                               Kokkos::View<double***> E,
                               Grid const & grid) 
    {
        ghostExchange(rho, rhou, E, grid); 
        bcUpdate(rho, rhou, E, grid); 
    }

private:    
    Buffer sbuf, rbuf;

};

class NullGradient : public IBoundaryCondition
{
public:
    NullGradient(Grid const & grid) : IBoundaryCondition(grid){};
    
    void bcUpdate(Kokkos::View<double***> rho,
                       Kokkos::View<double****> rhou,
                       Kokkos::View<double***> E,
                       Grid const & grid) const final
    {
        assert(rho.extent(0) == rhou.extent(0));
        assert(rhou.extent(0) == E.extent(0));
        assert(rho.extent(1) == rhou.extent(1));
        assert(rhou.extent(1) == E.extent(1));
        assert(rho.extent(2) == rhou.extent(2));
        assert(rhou.extent(2) == E.extent(2));

        int ng=0;
        int size_x=0;
        if(grid.is_border[0][0])
        {
            ng = grid.Nghost[0];
            Kokkos::parallel_for("NullGradient_implementation",
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
        if(grid.Ndim>=2)
        {
            ng = grid.Nghost[1];
            if(grid.is_border[1][0])
            {
                Kokkos::parallel_for("NullGradient_implementation",
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
            if(grid.is_border[1][1])
            {
                size_x = rho.extent(1)-1;
                Kokkos::parallel_for("NullGradient_implementation",
                                     Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                                     {0, 0, 0},
                                     {rho.extent_int(0), ng, rho.extent_int(2)}),
                                     KOKKOS_LAMBDA(int i, int j, int k)
                {
                    rho(i, size_x-j, k) = rho(i, size_x-ng, k);
                    for (int n=0; n<rhou.extent_int(3); n++)
                    {
                        rhou(i, size_x-j, k, n) = rhou(i, size_x-ng, k, n);
                    }
                    E(i, size_x-j, k) = E(i, size_x-ng, k);
                });
            } 
        }

        if(grid.Ndim==3)
        {
            ng = grid.Nghost[2];
            if(grid.is_border[2][0])
            {
                Kokkos::parallel_for("NullGradient_implementation",
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
            if(grid.is_border[2][1])
            {
                size_x = rho.extent(2)-1;
                Kokkos::parallel_for("NullGradient_implementation",
                                     Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                                     {0, 0, 0},
                                     {rho.extent_int(0), rho.extent_int(1), ng}),
                                     KOKKOS_LAMBDA(int i, int j, int k)
                {
                    rho(i, j, size_x-k) = rho(i, j, size_x-ng);
                    for (int n=0; n<rhou.extent_int(3); n++)
                    {    
                        rhou(i, j, size_x-k, n) = rhou(i, j, size_x-ng, n);
                    }
                    E(i, j, size_x-k) = E(i, j, size_x-ng);
                });
            } 
        }
    }
};

class PeriodicCondition : public IBoundaryCondition
{
public:

    PeriodicCondition(Grid const & grid) : IBoundaryCondition(grid){};
    
    void bcUpdate([[maybe_unused]]Kokkos::View<double***> rho,
                       [[maybe_unused]]Kokkos::View<double****> rhou,
                       [[maybe_unused]]Kokkos::View<double***> E,
                       [[maybe_unused]]Grid const & grid) const final
    {
        // do nothing
    }
};


class ReflexiveCondition : public IBoundaryCondition
{
public:
    ReflexiveCondition(Grid const & grid) : IBoundaryCondition(grid){};
    
    void bcUpdate(Kokkos::View<double***> rho,
                       Kokkos::View<double****> rhou,
                       Kokkos::View<double***> E,
                       Grid const & grid) const final
    {
        int ng=0;
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
            int offset=2*(rho.extent(0)-ng)-1;
            Kokkos::parallel_for("Reflexive_implementation",
                                 Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                                 {rho.extent_int(0)-ng, 0, 0},
                                 {rho.extent_int(0), rho.extent_int(1), rho.extent_int(2)}),
                                 KOKKOS_LAMBDA(int i, int j, int k)
            {
                rho(i, j, k) = rho(offset-i, j, k);
                for (int n=0; n<rhou.extent_int(3); n++)
                {    
                    rhou(i, j, k, n) = (n==grid.Ndim-1? -rhou(offset-i, j, k, n):rhou(offset-i, j, k, n));
                }
                E(i, j, k) = E(offset-i, j, k);
            });
        }
        if(grid.Ndim>=2)
        {
            ng = grid.Nghost[1];
            if(grid.is_border[1][0])
            {
                Kokkos::parallel_for("Reflexive_implementation",
                                     Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                                     {0, 0, 0},
                                     {rho.extent_int(0), ng, rho.extent_int(2)}),
                                     KOKKOS_LAMBDA(int i, int j, int k)
                {
                    rho(i, j, k) = rho(i, 2*ng-1-j, k);
                    for (int n=0; n<rhou.extent_int(3); n++)
                    {    
                        rhou(i, j, k, n) = (n==grid.Ndim-1? -rhou(i, 2*ng-1-j, k, n):rhou(i, 2*ng-1-j, k, n));
                    }
                    E(i, j, k) = E(i, 2*ng-1-j, k);
                });
            }
            if(grid.is_border[1][1])
            {
                int offset=2*(rho.extent(1)-ng)-1;
                Kokkos::parallel_for("Reflexive_implementation",
                                     Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                                     {0, rho.extent_int(1)-ng, 0},
                                     {rho.extent_int(0), rho.extent_int(1), rho.extent_int(2)}),
                                     KOKKOS_LAMBDA(int i, int j, int k)
                {
                    rho(i, j, k) = rho(i, offset-j, k);
                    for (int n=0; n<rhou.extent_int(3); n++)
                    {    
                        rhou(i, j, k, n) = (n==grid.Ndim-1? -rhou(i, offset-j, k, n):rhou(i, offset-j, k, n));
                    }
                    E(i, j, k) = E(i, offset-j, k);
                });
            }
        }
        if(grid.Ndim==3)
        {
            ng = grid.Nghost[2];
            if(grid.is_border[2][0])
            {
                Kokkos::parallel_for("Reflexive_implementation",
                                     Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                                     {0, 0, 0},
                                     {rho.extent_int(0), rho.extent_int(1), ng}),
                                     KOKKOS_LAMBDA(int i, int j, int k)
                {
                    rho(i, j, k) = rho(i, j, 2*ng-1-k);
                    for (int n=0; n<rhou.extent_int(3); n++)
                    {    
                        rhou(i, j, k, n) = (n==grid.Ndim-1? -rhou(i, j, 2*ng-1-k, n):rhou(i, j, 2*ng-1-k, n));
                    }
                    E(i, j, k) = E(i, j, 2*ng-1-k);
                });
            }
            if(grid.is_border[2][1])
            {
                int offset=2*(rho.extent(2)-ng)-1;
                Kokkos::parallel_for("Reflexive_implementation",
                                     Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                                     {0, 0, rho.extent_int(2)-ng},
                                     {rho.extent_int(0), rho.extent_int(1), rho.extent_int(2)}),
                                     KOKKOS_LAMBDA(int i, int j, int k)
                {
                    rho(i, j, k) = rho(i, j, offset-k);
                    for (int n=0; n<rhou.extent_int(3); n++)
                    {    
                        rhou(i, j, k, n) = (n==grid.Ndim-1? -rhou(i, j, offset-k, n):rhou(i, j, offset-k, n));
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
    xPeriodicyReflexiveCondition(Grid const & grid) : IBoundaryCondition(grid){};
    
    void bcUpdate(Kokkos::View<double***> rho,
                  Kokkos::View<double****> rhou,
                  Kokkos::View<double***> E,
                  Grid const & grid) const final
    {
        int ng=0;
        if(grid.Ndim>=2)
        {
            ng = grid.Nghost[1];
            if(grid.is_border[1][0])
            {
                Kokkos::parallel_for("Reflexive_implementation",
                                     Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                                     {0, 0, 0},
                                     {rho.extent_int(0), ng, rho.extent_int(2)}),
                                     KOKKOS_LAMBDA(int i, int j, int k)
                {
                    rho(i, j, k) = rho(i, 2*ng-1-j, k);
                    for (int n=0; n<rhou.extent_int(3); n++)
                    {    
                        rhou(i, j, k, n) = (n==grid.Ndim-1? -rhou(i, 2*ng-1-j, k, n):rhou(i, 2*ng-1-j, k, n));
                    }
                    E(i, j, k) = E(i, 2*ng-1-j, k);
                });
            }
            if(grid.is_border[1][1])
            {
                int offset=2*(rho.extent(1)-ng)-1;
                Kokkos::parallel_for("Reflexive_implementation",
                                     Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                                     {0, rho.extent_int(1)-ng, 0},
                                     {rho.extent_int(0), rho.extent_int(1), rho.extent_int(2)}),
                                     KOKKOS_LAMBDA(int i, int j, int k)
                {
                    rho(i, j, k) = rho(i, offset-j, k);
                    for (int n=0; n<rhou.extent_int(3); n++)
                    {    
                        rhou(i, j, k, n) = (n==grid.Ndim-1? -rhou(i, offset-j, k, n):rhou(i, offset-j, k, n));
                    }
                    E(i, j, k) = E(i, offset-j, k);
                });
            }
        }
    }
};

inline std::unique_ptr<IBoundaryCondition> factory_boundary_construction(
    Grid const & grid,
    std::string const& s)
{
    if (s == "NullGradient")
    {
        return std::make_unique<NullGradient>(grid);
    }
    if (s == "Periodic")
    {
        return std::make_unique<PeriodicCondition>(grid);
    }
    if (s == "Reflexive")
    {
        return std::make_unique<ReflexiveCondition>(grid);
    }
    if (s == "xPeriodicyReflexive")
    {
        return std::make_unique<xPeriodicyReflexiveCondition>(grid);
    }

    throw std::runtime_error("Unknown boundary condition : " + s + ".");
}
