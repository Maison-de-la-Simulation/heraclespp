/**
 * @file domain_dec.cpp
 * Code de calcul du decoupage au niveau MPI
 *  */
#include <iostream> 
#include <cmath>
#include <array>
#include <string>
#include <vector>
#include "grid.hpp"
#include <mpi.h>

using namespace std;

/* ****************************************************************
This routine compute the domaine partitioning
Npcu    : Number of cpu, input
MyRank  : Local process number
G       : Grid structure 
Ncpu_x  : Number of cpu along each direction, input
          Ncpu = Ncpu_x[0] * Ncpu_x[1] * Ncpu_x[2]  
*/
void Domain_partitioning (int Ncpu ,   int* Ncpu_x , Grid *G){

std::vector<int> lim_x(Ncpu_x[0]+1,1) ;
std::vector<int> lim_y(Ncpu_x[1]+1,1) ;
std::vector<int> lim_z(Ncpu_x[2]+1,1) ;

int ic   = 0 ;
int idim = 0 ;
int ix ;


// splitting in the x direction
  idim     = 0 ;
  lim_x[0] = 0 ; 
   for (ix=1; ix<Ncpu_x[idim]+1;ix++) {
     ic = G->Nx_glob[idim]/Ncpu_x[idim] ;
     if (ix <= (G->Nx_glob[idim]%Ncpu_x[idim])) {
        ic = ic+1 ; 
     }
     lim_x[ix] = lim_x[ix-1] + ic ;    
    }

// splitting in the y direction
  idim     = 1 ;
  lim_y[0] = 0 ; 
  for (int ix=1; ix<Ncpu_x[idim]+1;ix++) {
     ic = G->Nx_glob[idim]/Ncpu_x[idim] ;
     if (ix <= (G->Nx_glob[idim]%Ncpu_x[idim])) {
        ic = ic+1 ; 
     }
     lim_y[ix] = lim_y[ix-1] + ic ;    
    }
// splitting in the z direction
  idim     = 2 ;
  lim_z[0] = 0 ; 
  for (int ix=1; ix<Ncpu_x[idim]+1;ix++) {
     ic = G->Nx_glob[idim]/Ncpu_x[idim] ;
     if (ix <= (G->Nx_glob[idim]%Ncpu_x[idim])) {
        ic = ic+1 ; 
     }
     lim_z[ix] = lim_z[ix-1] + ic ;     
    } 

  int MyRank = 0;
  #ifdef ENABLE_MPI
   MPI_Comm_rank(MPI_COMM_WORLD , &MyRank  ) ;
  #endif

  int MyRank_cart[3] = {0,0,0} ;
  
  #ifdef ENABLE_MPI
  int Periodic[3]    = {1,1,1} ;
  MPI_Comm COMM_CART ;  
  MPI_Cart_create(MPI_COMM_WORLD, 3, Ncpu_x, Periodic, 1, &COMM_CART ) ;
  MPI_Cart_coords(COMM_CART , MyRank, 3, MyRank_cart ) ;
  #endif

  G->Corner_min[0] = lim_x[MyRank_cart[0]] ;
  G->Corner_min[1] = lim_y[MyRank_cart[1]] ;
  G->Corner_min[2] = lim_z[MyRank_cart[2]] ;

  G->Corner_max[0] = lim_x[MyRank_cart[0]+1] - 1 ;
  G->Corner_max[1] = lim_y[MyRank_cart[1]+1] - 1 ;
  G->Corner_max[2] = lim_z[MyRank_cart[2]+1] - 1 ;
  
  G->Nx_size[0] = G->Corner_max[0] - G->Corner_min[0] +1 + 2*G->Nghost ;
  G->Nx_size[1] = G->Corner_max[1] - G->Corner_min[1] +1 + 2*G->Nghost ;
  G->Nx_size[2] = G->Corner_max[2] - G->Corner_min[2] +1 + 2*G->Nghost ;

  cout << " MyRank = " << MyRank << " ix, iy, iz " << MyRank_cart[0] << " " << MyRank_cart[1]<< " " << MyRank_cart[2] << endl ;
  cout << " MyRank = " << MyRank << " imin, imax " << G->Corner_min[0] << " " << G->Corner_max[0] << endl;
  cout << " MyRank = " << MyRank << " jmin, jmax " << G->Corner_min[1] << " " << G->Corner_max[1] << endl;
  cout << " MyRank = " << MyRank << " kmin, kmax " << G->Corner_min[2] << " " << G->Corner_max[2] << endl;

  return ;  
}


/* ****************************************************************
This routine distribute the cpu over the various direction in an optimum way
Npcu    : Number of cpu, input
Nx_glob : Number of cells along each direction, input
Ndim    : Number of dimension, input

Ncpu_x  : Number of cpu along each direction, output
          Ncpu = Ncpu_x[0] * Ncpu_x[1] * Ncpu_x[2]  
*/

void Cpu_Dec (int Ncpu , int Ndim, array<int,3> Nx_glob , int* Ncpu_x) {

constexpr int Pmax = 5000 ; //Largest possible prime number considered

std::vector<int> PrimeNumber(Pmax, 1)  ; // List of prime numbers
std::vector<int> Number     (Pmax, 1)  ; 

for(int i=0;i<Pmax ;i=i+1) { // Initialisation de Number à 1
    Number[i] = 1 ;
}

for(int i=2;i<Pmax ;i=i+1) { 
    if (Number[i] == 1) {
        for(int j=2*i;j<Pmax;j=j+i) {
            Number[j] = 0 ;
        }   
    }
}

int j = 0 ;
for(int i=1;i<Pmax ;i=i+1) {     
    if(Number[i] == 1) {
        j=j+1 ;
        PrimeNumber[j] = i ;
    }
}

// for(int i=0;i<Nprime+1 ;i=i+1) cout << i << "  " << PrimeNumber[i] << endl ;

/* Decomposition of Ncpu in Prime Number */

std::vector<int> Dec(100, 0) ; // Storage for the prime number decomposition

if (Ncpu > Pmax) {
    cout << "Ncpu to large for prime factor decomposition (increase Pmax)" << endl ;
    cout << "Ncpu    = " << Ncpu << endl ;
    cout << "Pmax    = " << Pmax << endl ;
    /* Ajouter une fermeture propre du program*/
    return ; 
}

  int m  = Ncpu ;
  int i  = 2 ;
  int Np = 0 ; // Nomber of prime factor in the decomposition of Ncpu 
  while(m > 1) {
     if (m%PrimeNumber[i] == 0 ) {
        Np = Np + 1 ;
        Dec[Np] = PrimeNumber[i] ;
        m       =  m/PrimeNumber[i] ;
     }
     else {
        i = i+1 ;
     }
  }


/* Finding the best arrangement to distribute cpus among dimensions */ 

int Ncombi = std::pow(Ndim,Np) ; // Number of possible arrangement
int dnmin  = 1000000 ; 
  
/* integer, dimension(Ndim ) :: Cpu_Dim,NN,Cpu_dimt */
std::vector<int> Cpu_dim(3,1)  ; // Number of Cpu along each dimension
std::vector<int> Cpu_dimt(3,1) ; // Number of Cpu along each dimension (temporary array)

int nC    = 0 ;
int ppmax = 0 ;
int ppmin = 0 ;
int pp    = 0 ;
int dn    = 0 ;

for(i=0;i<Ncombi ;i=i+1) {   
    m = i ;
    Cpu_dimt[0] = 1 ;  Cpu_dimt[1] = 1 ;  Cpu_dimt[2] = 1 ; 
    for(j=1;j<Np+1 ;j=j+1) {   
        nC = m/std::pow(Ndim, (Np-j))        ;
        m  = m - nC * std::pow(Ndim,(Np-j))  ;
        Cpu_dimt[nC] =  Cpu_dimt[nC] * Dec[j] ;
    }
     
    ppmax = 0 ;
    ppmin = 10000000 ; 
    for(j=0;j<Ndim ;j=j+1) {   
        pp    = Nx_glob[j]/Cpu_dimt[j] ;
        ppmax = std::max(pp,ppmax) ;
        ppmin = std::min(pp,ppmin) ;
    }
     
    dn = ppmax-ppmin ;
    if (dn < dnmin) {
        dnmin   = dn ;
        Cpu_dim[0] = Cpu_dimt[0] ; Cpu_dim[1] = Cpu_dimt[1] ; Cpu_dim[2] = Cpu_dimt[2] ;
    }

}

Ncpu_x[0] = Cpu_dim[0] ; Ncpu_x[1] = Cpu_dim[1] ; Ncpu_x[2] = Cpu_dim[2] ;

}


