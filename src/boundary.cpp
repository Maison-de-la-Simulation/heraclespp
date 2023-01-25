#include <Kokkos_Core.hpp>

#include "boundary.hpp"
#include "grid.hpp"

void GradientNull(
    Kokkos::View<double ***> const rho,
    Kokkos::View<double ***> const rhou,
    Kokkos::View<double ***> const E)
{
    Grid grid;
    int size = rho.extent(0) - 2*grid.Nghost;
    Kokkos::parallel_for(
        "boundary",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
        {0, 0, 0},
        {2, 1, 1}),
        KOKKOS_LAMBDA(int i, int j, int k)
    {
        rho(i, j, k) = rho(2, j, k);    
        rho(size+2+i, j, k) = rho(size+1, j, k);

        rhou(i, j, k) = rhou(2, j, k);    
        rhou(size+2+i, j, k) = rhou(size+1, j, k);

        E(i, j, k) = E(2, j, k);    
        E(size+2+i, j, k) = E(size+1, j, k); 
    });
}

void Periodic(
    Kokkos::View<double ***> const rho,
    Kokkos::View<double ***> const rhou,
    Kokkos::View<double ***> const E)
{
    Grid grid;
    int size = rho.extent(0) - 2*grid.Nghost;
    Kokkos::parallel_for(
        "boundary",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
        {0, 0, 0},
        {2, 1, 1}),
        KOKKOS_LAMBDA(int i, int j, int k)
    {
        rho(i, j, k) = rho(size+i, j, k);    
        rho(size+2+i, j, k) = rho(2+i, j, k);

        rhou(i, j, k) = rhou(size+i, j, k);    
        rhou(size+2+i, j, k) = rhou(2+i, j, k);

        E(i, j, k) = E(size+i, j, k);    
        E(size+2+i, j, k) = E(2+i, j, k);
    });
}

void Reflexive(
    Kokkos::View<double ***> const rho,
    Kokkos::View<double ***> const rhou,
    Kokkos::View<double ***> const E)
{
    Grid grid;
    int size = rho.extent(0) - 2*grid.Nghost;
    Kokkos::parallel_for(
        "boundary",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
        {0, 0, 0},
        {2, 1, 1}),
        KOKKOS_LAMBDA(int i, int j, int k)
    {
        rho(i, j, k) = rho(3-i, j, k);    
        rho(size+2+i, j, k) = rho(size+1, j, k);

        rhou(i, j, k) = - rhou(3-i, j, k);    
        rhou(size+2+i, j, k) = - rhou(size+1, j, k);

        E(i, j, k) = E(3-i, j, k);    
        E(size+2+i, j, k) = E(size+1, j, k);
    });
}

/*
 if(bord=='trans'):
            # Conditions aux limites : transmitives
            U[:,0] = U[:,2]
            U[:,1] = U[:,2]
            U[:,nx+2] = U[:,nx+1]
            U[:,nx+3] = U[:,nx+1]
        elif(bord=='per'):
            # Conditions aux limites : périodiques
            U[:,0] = U[:,nx]
            U[:,1] = U[:,nx+1]
            U[:,nx+2] = U[:,2]
            U[:,nx+3] = U[:,3]
        elif(bord=='ref'):
            # Conditions aux limites : réflexives
            U[:,0] = U[:,3]
            U[:,1] = U[:,2]
            U[:,nx+2] = U[:,nx+1]
            U[:,nx+3] = U[:,nx+1]
            # Vitesse négative
            U[1,0] = - U[1,3]
            U[1,1] = - U[1,2]
*/
