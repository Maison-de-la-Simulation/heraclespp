/**
 * @file units.hpp
 *  
 * Define code units and useful constants and units.
 * 
 * */
#pragma once
#include <cmath>

namespace Units{
    constexpr double Pi      = M_PI    ;
    constexpr double four_Pi = 4 * Pi  ;

// code units in SI (kg / metre / seconde) 
    constexpr double unitM = 1.0E+00 ; // Mass unit in kg
    constexpr double unitL = 1.0E+00 ; // Length unit in meter
    constexpr double unitT = 1.0E+00 ; // Time unit in second
    constexpr double unitD = unitM/(unitL*unitL*unitL) ; // Density unit

//  cgs in the new basis
    constexpr double gramme      = 1.0E-3/unitM ;
    constexpr double centimetre  = 1.0E-2/unitL ;
    constexpr double seconde     = 1.0   /unitT ;
    constexpr double degre       = 1.0          ;
    constexpr double gauss       = 1.0          ;
  
// additional units
    constexpr double cm2     =  1.000000e+00 * centimetre*centimetre ;
    constexpr double dyne    =  1.000000e+00 * gramme*centimetre/(seconde*seconde) ;
    constexpr double erg     =  1.000000e+00 * gramme*cm2/(seconde*seconde) ;  

// physical constants
    constexpr double c       =  2.997925e+10 * centimetre/seconde ;    
    constexpr double c2      =  c*c ;
    constexpr double eV      =  1.602200e-12 * erg ;
    constexpr double G       =  6.673000e-08 * dyne*cm2/(gramme*gramme) ;
    constexpr double hplanck =  6.626200e-27 * erg*seconde ;
    constexpr double kb      =  1.380620e-16 * erg/degre ;
    constexpr double uma     =  1.660531e-24 * gramme ;
    constexpr double a_R     = (8.0*(Pi*Pi*Pi*Pi*Pi)*(kb*kb*kb*kb))/(15.0*(hplanck*hplanck*hplanck)*(c2*c)) ;
    constexpr double Lsol    =  3.826800e+33 * erg/seconde ;
    constexpr double Msol    =  1.989100e+33 * gramme ;
    constexpr double Rsol    =  6.959900e+10 * centimetre ; 
    constexpr double pc      =  3.085678e+18 * centimetre ;
    constexpr double mu0     =  4.000000e+00 * Pi * gauss*gauss * centimetre * seconde*seconde / gramme ;
    constexpr double au      =  1.495980e+13 * centimetre ;
  
// useful units
    constexpr double cm3     =  1.000000e+00 * cm2*centimetre ;
    constexpr double ms      =  1.000000e+02 * centimetre/seconde ;
    constexpr double kms     =  1.000000e+05 * centimetre/seconde ;
    constexpr double year    =  3.155760e+07 * seconde ;
    constexpr double Myrs    =  3.155760e+13 * seconde ;
    constexpr double Gyrs    =  3.155760e+16 * seconde ;
    constexpr double kpc     =  3.085678e+21 * centimetre ;
    constexpr double Mpc     =  3.085678e+24 * centimetre ;
    constexpr double Kelvin  =  1.000000e+00 * degre ;
    constexpr double kg      =  1.000000e+03 * gramme ;
    constexpr double metre   =  1.000000e+02 * centimetre ;
    constexpr double Pascal  =  1.000000e+00 * kg/metre      /(seconde*seconde) ;
    constexpr double Joule   =  1.000000e+00 * kg*metre*metre/(seconde*seconde) ;
    constexpr double H0      =  7.200000e+01 * kms/Mpc ;
}