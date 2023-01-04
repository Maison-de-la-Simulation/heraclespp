#include <iostream>
#include <Kokkos_Core.hpp>

/*! @fn ShockTubeInit (double rho, double rhou, double E, int inter) */

/**<Initialisation for the schock tube problem */
/**< Input : */
/**< rho   : array(nx) : density table */
/**< rhou  : array(nx) : momentum table */
/**< E     : array(nx) : energy table */
/**< inter : int       : interface value */

/**< Output: */
/**< rho   : array(nx) : density table */
/**< rhou  : array(nx) : momentum table */
/**< E     : array(nx) : energy table */

void ShockTubeInit(Kokkos::View<double*> rho, Kokkos::View<double*> rhou, Kokkos::View<double*> E, int inter, int nx){
  // Left side
  double const rhoL = 1; // Density
  double const uL   = 0; // Speed
  double const PL   = 1; // Pressure
  // Right side
  double const rhoR = 0.125;
  double const uR   = 0;
  double const PR   = 0.1;

  Kokkos::parallel_for("Init", nx, KOKKOS_LAMBDA(int i){
    rho(i) = rhoL;
    //printf("Greeting from iteration %i\n",i);
  });

}
