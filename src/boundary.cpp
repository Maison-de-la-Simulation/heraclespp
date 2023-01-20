#include <Kokkos_Core.hpp>

#include "boundary.hpp"

void GradientNull(
    Kokkos::View<double *> const rho,
    Kokkos::View<double *> const rhou,
    Kokkos::View<double *> const E,
    int size)
{
    rho(0) = rho(1)= rho(2);
    rhou(0) = rhou(1)= rhou(2);
    E(0) = E(1)= E(2);

    rho(size+2) = rho(size+3) = rho(size+1); 
    rhou(size+2) = rhou(size+3) = rhou(size+1); 
    E(size+2) = E(size+3) = E(size+1); 
}

/*
 if(bord=='trans'):
            # Conditions aux limites : transmitives
            U_new[:,0] = U_new[:,2]
            U_new[:,1] = U_new[:,2]
            U_new[:,nx+2] = U_new[:,nx+1]
            U_new[:,nx+3] = U_new[:,nx+1]
        elif(bord=='per'):
            # Conditions aux limites : périodiques
            U_new[:,0] = U_new[:,nx]
            U_new[:,1] = U_new[:,nx+1]
            U_new[:,nx+2] = U_new[:,2]
            U_new[:,nx+3] = U_new[:,3]
        elif(bord=='ref'):
            # Conditions aux limites : réflexives
            U_new[:,0] = U_new[:,3]
            U_new[:,1] = U_new[:,2]
            U_new[:,nx+2] = U_new[:,nx+1]
            U_new[:,nx+3] = U_new[:,nx+1]
            # Vitesse négative
            U_new[1,0] = - U_new[1,3]
            U_new[1,1] = - U_new[1,2]
*/