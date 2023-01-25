/**
 * @file globla_var.hpp
 * Variables with gamme dependance
 */

#pragma once

#include <cmath>

// Transformer en structure. Mettre dans un fichier EOS.
namespace GV{
    inline double gamma;
}

struct Grid {
int Ndim       ; // Number of dimension of the run 1-3 (default = 1)
int Nx_glob[3] ; // Total number of cells in each directions (excluding ghost)
int Nx[3]      ; // Number of cells on the local MPI process
int Nghost     ; // Number of ghost cells (default = 2)
int Nx_size[3] ; // Size of local arrays 
int NBlock[3]  ; // number of sub-blocks

// taille des tableaux N (0:N-1) tjrs

Grid(int Ncpu , int MyRank){
    Ndim     = 1 ;
    Nghost   = 2 ;
   
} 
} ; // end of Grid

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
