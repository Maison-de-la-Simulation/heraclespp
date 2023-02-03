/**
 * @file cfl_cond.hpp
 * CFL condition
 */

#pragma once

#include <Kokkos_Core.hpp>

namespace thermodynamics
{

class PerfectGas;

} // namespace thermodynamics

//! Time step with the cfl condition
//! @param[in] rho density array 3D
//! @param[in] u celerity array 3D
//! @param[in] P pressure array 3D
//! @param[in] dx space step in x
//! @param[in] dy space step in y
//! @param[in] dz space step in z
//! @param[out] dt time step
[[nodiscard]] double time_step(
    double const cfl,
    Kokkos::View<double***> rho,
    Kokkos::View<double***> u,
    Kokkos::View<double***> P,
    double const dx,
    double const dy,
    double const dz, 
    thermodynamics::PerfectGas const& eos);
