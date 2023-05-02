//!
//! @file units.hpp
//! Define code units and useful constants and units.
//!

#pragma once

#include <Kokkos_Core.hpp>

namespace novapp::units
{
// code units in SI : kg / meter / seconde
    constexpr double unit_M = 1.0; // Mass : kilogram
    constexpr double unit_L = 1.0; // Length : meter
    constexpr double unit_T = 1.0; // Time : second

// SI units
    constexpr double kg = 1.0;
    constexpr double m = 1.0;
    constexpr double s = 1.0;
    constexpr double Kelvin = 1.0;

    constexpr double kg2 = kg * kg;
    constexpr double m2 = m * m;
    constexpr double m3 = m2 * m;
    constexpr double s2 = s * s;
    constexpr double K4 = Kelvin * Kelvin * Kelvin * Kelvin;

    constexpr double density = kg / m3; // kg.m^{-3}
    constexpr double velocity = m / s;
    constexpr double acc = m / s2; // Acceleration : m.s^{-2}
    constexpr double pressure = kg / (m * s2);
    constexpr double Joule = kg * m2 / s2; // kg.m^{2}.s^{-2}
    constexpr double Newton = kg * m / s2; // kg.m.s^{-2}
    constexpr double Watt = Joule / s; // kg.m^{2}.s^{-3}
    constexpr double evol = Joule / m3; // Volumic energy : kg.m^{-1}.s^{-2}
    constexpr double espe = Joule / kg; // Specific energy : m^{2}.s^{-2}

//  cgs in the new basis
    constexpr double g = 1.0E-3 / kg;
    constexpr double cm = 1.0E-2 / m;

    constexpr double cm2 = cm * cm;
    constexpr double cm3 = cm2 * cm;

    constexpr double dens_cgs = g / cm3; // Density
    constexpr double cm_s = cm / s; // Velocity : cm.s^{-1}
    constexpr double acc_cgs = cm / s2; // Acceleration : cm.s^{-2}
    constexpr double press_cgs = g / (cm * s2); // Pressure
    constexpr double erg = g * cm2 / s2; // Energy : g.cm^{2}.s^{-2}
    constexpr double dyne = g * cm / s2; // Dyne : g.cm.s^{-2}
    constexpr double evol_cgs = erg / cm3; // Volumic energy : g.cm^{-1}.s^{-2}
    constexpr double espe_cgs = erg / g; // Specific energy : cm^{2}.s^{-2}

// physical constants : SI
    constexpr double kb = 1.38E-23 * Joule / Kelvin; // kg.m^{2}.s^{-2}.K^{-1}
    constexpr double mh = 1.67E-27 * kg; // kg
    constexpr double c = 2.99792458E8 * velocity; // m.s^{-1}
    constexpr double G = 6.6743015E-11 * Newton * m2 / kg2; // m^{3}.kg^{-1}.s^{-2}
    constexpr double hplanck = 6.62607015E-34 * Joule * s; // kg.m^{2}.s^{-3}
    constexpr double eV = 1.602176634E-19 * Joule; // kg.m^{2}.s^{-2}

    constexpr double pi = Kokkos::numbers::pi;
    constexpr double pi5 = pi * pi * pi * pi * pi;
    constexpr double kb4 = kb * kb * kb * kb;
    constexpr double hplanck3 = hplanck * hplanck * hplanck;
    constexpr double c2 = c * c;
    constexpr double ar = (8 * pi5 * kb4) / (15 * hplanck3 * c3) * Watt / (m2 * K4); // kg.m^{-1}.s^{-2}.K^{-4}

    constexpr double ua = 1.49597870E11* m;
    
    constexpr double Msol = 1.9885E30 * kg;
    constexpr double Rsol = 6.96342E8 * m; 
    constexpr double Lsol = 3.8268E26 * Watt;

// useful units
    constexpr double year = 3.155760E7 * s;
    constexpr double Myrs = 3.155760E13 * s;
    constexpr double Gyrs = 3.155760E16 * s;
    constexpr double pc = 3.085677581E16 * m;
    constexpr double kpc = pc * 1E3;
    constexpr double Mpc = kpc * 1E3;
    constexpr double Pascal = kg / (m * s2); // kg.m^{-1}.s^{-1}
    constexpr double H0 = 7.2E1 * m * 1E3 / Mpc; // cste de Hubble

} // namespace novapp::units
