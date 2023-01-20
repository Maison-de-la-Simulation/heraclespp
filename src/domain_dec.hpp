/**
 * @file domain_dec.hpp
 * Code de calcul du decoupage au niveau MPI
 *  */
#pragma once

#include <iostream> 
#include <cmath>
#include <array>
#include <string>
#include <vector>
#include "grid.hpp"
#include <mpi.h>

/* ****************************************************************
This routine compute the domaine partitioning
Npcu    : Number of cpu, input
MyRank  : Local process number
G       : Grid structure 
Ncpu_x  : Number of cpu along each direction, input
          Ncpu = Ncpu_x[0] * Ncpu_x[1] * Ncpu_x[2]  
*/

void Domain_partitioning (int Ncpu ,   int* Ncpu_x , Grid *G) ;


/* ****************************************************************
This routine distribute the cpu over the various direction in an optimum way
Npcu    : Number of cpu, input
Nx_glob : Number of cells along each direction, input
Ndim    : Number of dimension, input

Ncpu_x  : Number of cpu along each direction, output
          Ncpu = Ncpu_x[0] * Ncpu_x[1] * Ncpu_x[2]  
*/

void Cpu_Dec (int Ncpu , int Ndim, array<int,3> Nx_glob , int* Ncpu_x) ;
