//!
//! @file euler_equations.hpp
//!

#pragma once

#include <Kokkos_Core.hpp>
#include <PerfectGas.hpp>

#include "kronecker.hpp"
#include "ndim.hpp"

namespace novapp
{

struct EulerCons
{
    double density;
    std::array<double, ndim> momentum;
    double energy;
};

struct EulerFlux
{
    double density;
    std::array<double, ndim> momentum;
    double energy;
};

struct EulerPrim
{
    double density;
    std::array<double, ndim> velocity;
    double pressure;
};

KOKKOS_INLINE_FUNCTION
double compute_volumic_kinetic_energy(EulerCons const& cons) noexcept
{
    double norm_momentum = 0;
    for (int idim = 0; idim < ndim; ++idim)
    {
        norm_momentum += cons.momentum[idim] * cons.momentum[idim];
    }
    return 0.5 * norm_momentum / cons.density;
}

KOKKOS_INLINE_FUNCTION
double compute_volumic_kinetic_energy(EulerPrim const& prim) noexcept
{
    double norm_velocity = 0;
    for (int idim = 0; idim < ndim; ++idim)
    {
        norm_velocity += prim.velocity[idim] * prim.velocity[idim];
    }
    return 0.5 * prim.density * norm_velocity;
}

//! Flux formula
//! @param[in] prim Primitive state
//! @param[in] eos Equation of state
//! @return flux
KOKKOS_INLINE_FUNCTION
EulerFlux compute_flux(
        EulerPrim const& prim,
        int locdim,
        thermodynamics::PerfectGas const& eos) noexcept
{
    EulerFlux flux;
    double const volumic_total_energy
            = compute_volumic_kinetic_energy(prim)
              + eos.compute_volumic_internal_energy(prim.density, prim.pressure);
    flux.density = prim.density * prim.velocity[locdim];
    for (int idim = 0; idim < ndim; ++idim)
    {
        flux.momentum[idim] = prim.density * prim.velocity[locdim] * prim.velocity[idim];
    }
    flux.momentum[locdim] += prim.pressure;
    flux.energy = prim.velocity[locdim] * (volumic_total_energy + prim.pressure);
    return flux;
}

//! Flux formula
//! @param[in] cons Conservative state
//! @param[in] eos Equation of state
//! @return flux
KOKKOS_INLINE_FUNCTION
EulerFlux compute_flux(
        EulerCons const& cons,
        int locdim,
        thermodynamics::PerfectGas const& eos) noexcept
{
    EulerFlux flux;
    double const volumic_internal_energy = cons.energy - compute_volumic_kinetic_energy(cons);
    double const pressure = eos.compute_pressure(cons.density, volumic_internal_energy);
    double const velocity = cons.momentum[locdim] / cons.density;
    flux.density = velocity * cons.density;
    for (int idim = 0; idim < ndim; ++idim)
    {
        flux.momentum[idim] = cons.momentum[locdim] * cons.momentum[idim] / cons.density;
    }
    flux.momentum[locdim] += pressure;
    flux.energy = velocity * (cons.energy + pressure);
    return flux;
}

KOKKOS_INLINE_FUNCTION
EulerPrim to_prim(EulerCons const& cons, thermodynamics::PerfectGas const& eos) noexcept
{
    EulerPrim prim;
    prim.density = cons.density;
    for (int idim = 0; idim < ndim; ++idim)
    {
        prim.velocity[idim] = cons.momentum[idim] / cons.density;
    }
    prim.pressure = eos.compute_pressure(
            cons.density,
            cons.energy - compute_volumic_kinetic_energy(cons));
    return prim;
}

KOKKOS_INLINE_FUNCTION
EulerCons to_cons(EulerPrim const& prim, thermodynamics::PerfectGas const& eos) noexcept
{
    EulerCons cons;
    cons.density = prim.density;
    for (int idim = 0; idim < ndim; ++idim)
    {
        cons.momentum[idim] = prim.density * prim.velocity[idim];
    }
    cons.energy = eos.compute_volumic_internal_energy(prim.density, prim.pressure)
                  + compute_volumic_kinetic_energy(prim);
    return cons;
}

} // namespace novapp
