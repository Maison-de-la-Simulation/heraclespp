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

    virtual void execute(
        Kokkos::View<double***> rho,
        Kokkos::View<double***> rhou,
        Kokkos::View<double***> E,
        int nghost_cell) const 
        = 0;
};

class NullGradient : public IBoundaryCondition
{
public:
    void execute(
        Kokkos::View<double***> const rho,
        Kokkos::View<double***> const rhou,
        Kokkos::View<double***> const E,
        int nghost_cell) const final
    {
        //static constexpr std::string_view s_label = "NullGradient";
        
        assert(rho.extent(0) == rhou.extent(0));
        assert(rhou.extent(0) == E.extent(0));
        assert(rho.extent(1) == rhou.extent(1));
        assert(rhou.extent(1) == E.extent(1));
        assert(rho.extent(2) == rhou.extent(2));
        assert(rhou.extent(2) == E.extent(2));

        int size_x = rho.extent(0) - 2*nghost_cell;

        Kokkos::parallel_for(
        "NullGradient_implementation",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
        {0, 0, 0},
        {nghost_cell, 1, 1}),
        KOKKOS_LAMBDA(int i, int j, int k)
        {
            rho(i, j, k) = rho(2, j, k);
            rho(size_x+2+i, j, k) = rho(size_x+1, j, k);

            rhou(i, j, k) = rhou(2, j, k);
            rhou(size_x+2+i, j, k) = rhou(size_x+1, j, k);

            E(i, j, k) = E(2, j, k);
            E(size_x+2+i, j, k) = E(size_x+1, j, k);
        });
    }
};

class PeriodicCondition : public IBoundaryCondition
{
public:
    void execute(
        Kokkos::View<double***> const rho,
        Kokkos::View<double***> const rhou,
        Kokkos::View<double***> const E,
        int nghost_cell) const final
    {
        assert(rho.extent(0) == rhou.extent(0));
        assert(rhou.extent(0) == E.extent(0));
        assert(rho.extent(1) == rhou.extent(1));
        assert(rhou.extent(1) == E.extent(1));
        assert(rho.extent(2) == rhou.extent(2));
        assert(rhou.extent(2) == E.extent(2));

        int size_x = rho.extent(0) - 2*nghost_cell;

        Kokkos::parallel_for(
        "Periodic_implementation",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
        {0, 0, 0},
        {nghost_cell, 1, 1}),
        KOKKOS_LAMBDA(int i, int j, int k)
    {
        rho(i, j, k) = rho(size_x+i, j, k);    
        rho(size_x+2+i, j, k) = rho(2+i, j, k);

        rhou(i, j, k) = rhou(size_x+i, j, k);    
        rhou(size_x+2+i, j, k) = rhou(2+i, j, k);

        E(i, j, k) = E(size_x+i, j, k);    
        E(size_x+2+i, j, k) = E(2+i, j, k);
    });
    }
};

class ReflexiveCondition : public IBoundaryCondition
{
public:
    void execute(
        Kokkos::View<double***> const rho,
        Kokkos::View<double***> const rhou,
        Kokkos::View<double***> const E,
        int nghost_cell) const final
    {
        assert(rho.extent(0) == rhou.extent(0));
        assert(rhou.extent(0) == E.extent(0));
        assert(rho.extent(1) == rhou.extent(1));
        assert(rhou.extent(1) == E.extent(1));
        assert(rho.extent(2) == rhou.extent(2));
        assert(rhou.extent(2) == E.extent(2));

        int size_x = rho.extent(0) - 2*nghost_cell;

        Kokkos::parallel_for(
        "Reflexive_implementation",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
        {0, 0, 0},
        {nghost_cell, 1, 1}),
        KOKKOS_LAMBDA(int i, int j, int k)
    {
        rho(i, j, k) = rho(3-i, j, k);    
        rho(size_x+2+i, j, k) = rho(size_x+1, j, k);

        rhou(i, j, k) = - rhou(3-i, j, k);    
        rhou(size_x+2+i, j, k) = - rhou(size_x+1, j, k);

        E(i, j, k) = E(3-i, j, k);    
        E(size_x+2+i, j, k) = E(size_x+1, j, k);
    });
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