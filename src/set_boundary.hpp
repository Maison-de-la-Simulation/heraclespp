//!
//! @file set_boundary.hpp
//!
#pragma once

#include <Kokkos_Core.hpp>

class IBoundaryCondition
{
public:
    IBoundaryCondition() = default;

    IBoundaryCondition(IBoundaryCondition const& x) = default;

    IBoundaryCondition(IBoundaryCondition&& x) noexcept = default;

    virtual ~IBoundaryCondition() noexcept = default;

    IBoundaryCondition& operator=(IBoundaryCondition const& x) = default;

    IBoundaryCondition& operator=(IBoundaryCondition&& x) noexcept = default;

    virtual void outerExchange(Kokkos::View<double***> rho,
                               Kokkos::View<double****> rhou,
                               Kokkos::View<double***> E,
                               Grid *grid) const = 0;

};

class NullGradient : public IBoundaryCondition
{
public:
    
    void outerExchange(Kokkos::View<double***> rho,
                       Kokkos::View<double****> rhou,
                       Kokkos::View<double***> E,
                       Grid *grid) const final
    {
        assert(rho.extent(0) == rhou.extent(0));
        assert(rhou.extent(0) == E.extent(0));
        assert(rho.extent(1) == rhou.extent(1));
        assert(rhou.extent(1) == E.extent(1));
        assert(rho.extent(2) == rhou.extent(2));
        assert(rhou.extent(2) == E.extent(2));

        int ng=0;
        int size_x=0;
        if(grid->mpi_rank_cart[0]==0)
        {
            ng = grid->Nghost[0];
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
        if(grid->mpi_rank_cart[0] == grid->Ncpu_x[0]-1)
        {
            ng = grid->Nghost[0];
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
        if(grid->Ndim>=2)
        {
            ng = grid->Nghost[1];
            if(grid->mpi_rank_cart[1]==0)
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
            if(grid->mpi_rank_cart[1] == grid->Ncpu_x[1]-1)
            {
                size_x = rho.extent(1)-1;
                Kokkos::parallel_for("NullGradient_implementation",
                                     Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                                     {0, rho.extent_int(1)-ng, 0},
                                     {rho.extent_int(0), rho.extent_int(1), rho.extent_int(2)}),
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
        if(grid->Ndim==3)
        {
            ng = grid->Nghost[2];
            if(grid->mpi_rank_cart[1]==0)
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
            if(grid->mpi_rank_cart[2] == grid->Ncpu_x[2]-1)
            {
                size_x = rho.extent(2)-1;
                Kokkos::parallel_for("NullGradient_implementation",
                                     Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                                     {0, 0, rho.extent_int(2)-ng},
                                     {rho.extent_int(0), rho.extent_int(1), rho.extent_int(2)}),
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
    
    void outerExchange([[maybe_unused]]Kokkos::View<double***> rho,
                       [[maybe_unused]]Kokkos::View<double****> rhou,
                       [[maybe_unused]]Kokkos::View<double***> E,
                       [[maybe_unused]]Grid *grid) const final
    {
        // do nothing
    }
};


class ReflexiveCondition : public IBoundaryCondition
{
public:
    
    void outerExchange(Kokkos::View<double***> rho,
                       Kokkos::View<double****> rhou,
                       Kokkos::View<double***> E,
                       Grid *grid) const final
    {
        int ng=0;
        if(grid->mpi_rank_cart[0]==0)
        {
            ng = grid->Nghost[0];
            Kokkos::parallel_for("Reflexive_implementation",
                                 Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                                 {0, 0, 0},
                                 {ng, rho.extent_int(1), rho.extent_int(2)}),
                                 KOKKOS_LAMBDA(int i, int j, int k)
            {
                rho(i, j, k) = rho(2*ng-1-i, j, k);
                for (int n=0; n<rhou.extent_int(3); n++)
                {    
                    rhou(i, j, k, n) = rhou(2*ng-1-i, j, k, n);
                }
                E(i, j, k) = E(2*ng-1-i, j, k);
            });
        }
        if(grid->mpi_rank_cart[0] == grid->Ncpu_x[0]-1)
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
                    rhou(i, j, k, n) = - rhou(offset-i, j, k, n);
                }
                E(i, j, k) = E(offset-i, j, k);
            });
        }
        if(grid->Ndim>=2)
        {
            ng = grid->Nghost[1];
            if(grid->mpi_rank_cart[1]==0)
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
                        rhou(i, j, k, n) = rhou(i, 2*ng-1-j, k, n);
                    }
                    E(i, j, k) = E(i, 2*ng-1-j, k);
                });
            }
            if(grid->mpi_rank_cart[1] == grid->Ncpu_x[1]-1)
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
                        rhou(i, j, k, n) = -rhou(i, offset-j, k, n);
                    }
                    E(i, j, k) = E(i, offset-j, k);
                });
            }
        }
        if(grid->Ndim==3)
        {
            ng = grid->Nghost[2];
            if(grid->mpi_rank_cart[2]==0)
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
                        rhou(i, j, k, n) = -rhou(i, j, 2*ng-1-k, n);
                    }
                    E(i, j, k) = E(i, j, 2*ng-1-k);
                });
            }
            if(grid->mpi_rank_cart[2] == grid->Ncpu_x[2]-1)
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
                        rhou(i, j, k, n) = -rhou(i, j, offset-k, n);
                    }
                    E(i, j, k) = E(i, j, offset-k);
                });
            }
        }
    }
};

inline std::unique_ptr<IBoundaryCondition> factory_boundary_construction(
    std::string const& s)
{
    if (s == "NullGradient")
    {
        return std::make_unique<NullGradient>();
    }
    if (s == "Periodic")
    {
        return std::make_unique<PeriodicCondition>();
    }
    if (s == "Reflexive")
    {
        return std::make_unique<ReflexiveCondition>();
    }

    throw std::runtime_error("Unknown boundary condition : " + s + ".");
}
