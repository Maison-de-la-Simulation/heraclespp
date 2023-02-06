#pragma once

#include <Kokkos_Core.hpp>
#include <PerfectGas.hpp>

struct EulerCons
{
    double density;
    double momentum;
    double energy;
};

struct EulerFlux
{
    double density;
    double momentum;
    double energy;
};

struct EulerPrim
{
    double density;
    double velocity;
    double pressure;
};

inline double compute_volumic_kinetic_energy(EulerCons const& cons) noexcept
{
    return 0.5 * cons.momentum * cons.momentum / cons.density;
}

inline double compute_volumic_kinetic_energy(EulerPrim const& prim) noexcept
{
    return 0.5 * prim.density * prim.velocity * prim.velocity;
}

//! Flux formula
//! @param[in] prim Primitive state
//! @param[in] eos Equation of state
//! @return flux
inline EulerFlux compute_flux(EulerPrim const& prim, thermodynamics::PerfectGas const& eos) noexcept
{
    EulerFlux flux;
    double const volumic_total_energy
            = compute_volumic_kinetic_energy(prim)
              + eos.compute_volumic_internal_energy(prim.density, prim.pressure);
    flux.density = prim.density * prim.velocity;
    flux.momentum = prim.density * prim.velocity * prim.velocity + prim.pressure;
    flux.energy = prim.velocity * (volumic_total_energy + prim.pressure);
    return flux;
}

//! Flux formula
//! @param[in] cons Conservative state
//! @param[in] eos Equation of state
//! @return flux
inline EulerFlux compute_flux(EulerCons const& cons, thermodynamics::PerfectGas const& eos) noexcept
{
    EulerFlux flux;
    double const volumic_internal_energy = cons.energy - compute_volumic_kinetic_energy(cons);
    double const pressure = eos.compute_pressure(cons.density, volumic_internal_energy);
    double const velocity = cons.momentum / cons.density;
    flux.density = velocity * cons.density;
    flux.momentum = velocity * cons.momentum + pressure;
    flux.energy = velocity * (cons.energy + pressure);
    return flux;
}

inline EulerPrim to_prim(EulerCons const& cons, thermodynamics::PerfectGas const& eos) noexcept
{
    EulerPrim prim;
    prim.density = cons.density;
    prim.velocity = cons.momentum / cons.density;
    prim.pressure = eos.compute_pressure(
            cons.density,
            cons.energy - compute_volumic_kinetic_energy(cons));
    return prim;
}

inline EulerCons to_cons(EulerPrim const& prim, thermodynamics::PerfectGas const& eos) noexcept
{
    EulerCons cons;
    cons.density = prim.density;
    cons.momentum = prim.density * prim.velocity;
    cons.energy = eos.compute_volumic_internal_energy(prim.density, prim.pressure)
                  + compute_volumic_kinetic_energy(prim);
    return cons;
}
