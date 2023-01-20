/**
 * @file grid.hpp
 * Variables with gamme dependance
 */

#pragma once

#include <cmath>
#include <array>
#include <string>

struct Grid {
int                Ndim        ; // Number of dimension of the run 1-3 (default = 1)
std::array<int, 3> Nx_glob     ; // Total number of cells in each directions (excluding ghost)
std::array<int, 3> Nx          ; // Number of cells on the local MPI process
int                Nghost      ; // Number of ghost cells (default = 2)
std::array<int, 3> Nx_size     ; // Size of local arrays (including ghost cells) 
std::array<int, 3> NBlock      ; // number of sub-blocks (default (1,1,1))
std::array<int, 3> Nx_block    ; // Maximum size of sub-block, including ghost
std::array<int, 3> Corner_min  ; // Position of bottom/let  corner in the global grid
std::array<int, 3> Corner_max  ; // Position of upper/right corner in the global grid

std::array<std::string,6>  BC_glob ; // boudary type for the global domain 
std::array<std::string,6>  BC      ; // boudary type for the local  domain 

Grid(){
    Ndim     = 1 ; 
    Nghost   = 2 ; 
    
    Nx_glob[0] = 10 ;
    Nx_glob[1] = 1  ;
    Nx_glob[2] = 1  ; 

    Nx      = Nx_glob ; // default for a single MPI process
    Nghost  = 2       ; // default value

    Nx_size = Nx                         ;
    Nx_size[0] = Nx_size[0] + 2 * Nghost ; // Default for a 1D simulation

    NBlock[0] = 1   ; // Default is no sub-block 
    NBlock[1] = 1   ; // Default is no sub-block 
    NBlock[2] = 1   ; // Default is no sub-block 
    Nx_block = Nx_size ;

    //!    Type of boundary conditions possibilities are : 
    //!    "Internal", "Periodic", "Reflexive", NullGradient", UserDefined", "Null" (undefined) 
    BC_glob[0] = "Null" ; // No default value, must be defined bye the user.
    BC_glob[1] = "Null" ;
    BC_glob[2] = "Null" ;
    BC_glob[3] = "Null" ;
    BC_glob[4] = "Null" ;
    BC_glob[5] = "Null" ;

    BC = BC_glob ;
 }    

} ; // end of Grid

/*
struct Phys{
int Nvit       ; // Number of velocity component (default Ndim)
int Nvar       ; // Number of hydro/MHD variable
int Nvar_ray   ; // Number of radiative variables
int Nfx        ; // Number of passive scalar (default = 0)

Phys(int Ndim){
    Nvit     = Ndim ;
    Nvar     = 2 + Nvit ;
    Nfx      = 0 ; 
    Nvar_ray = 1 + Ndim ;
}
} ;
*/

/*
implicit none
  integer, dimension(:,:), allocatable :: nx_cpu  Kokkos view   !< Number of cells in the x,y,z directions per cpu
  integer, dimension(3)                :: nx_glob               !< Global simulation dimensions
  integer, dimension(3)                :: nx                    !< Local simulation dimensions
  integer, dimension(3)                :: nxmin                 !< Lower limit for arrays
  integer, dimension(3)                :: nxmax                 !< Upper limit for arrays
  integer                              :: ndim                  !< Number of dimensions
  integer                              :: nx_max                !< Highest nx
  integer                              :: nx_glob_max           !< Global highest nx
  integer                              :: Nbuf                  !< Number of buffer/ghost cells
    integer                              :: nvar_ray              !< Total number of radiative variables
  integer                              :: slope_type            !< Type of slope limiter for Hydro and MHD
  real   , parameter                   :: Pi = 3.1415926535898  !< \f$  \pi \f$
  real   , parameter                   :: quatre_Pi = 4.*pi     !< \f$ 4\pi \f$
  real   , parameter                   :: smallc = 1.d-30       !< Dimensioned small constant
  real   , parameter                   :: smallr = 1.d-40       !< Dimensioned small real number
  character(len=10)                    :: riemann               !< Type of Riemann solver
  character(len=10)                    :: riemann2d             !< Type of Riemann solver 2
  real                                 :: switch_solv_pre       !< Switch from hlld to llf if beta > switch_solv_pre
  real                                 :: switch_solv_dens      !< Switch from hlld to llf if delta_rho > switch_solv_pre
  logical                              :: pressure_fix          !< Flag for pressure fix
  real                                 :: eps_pf                !< ratio of internal energy to kinetic energy at which the code switch to pressure fix
*/