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
//! @param[in] dx space step array 3D
//! @return time step
[[nodiscard]] double time_step(
    double const cfl,
    Kokkos::View<const double***> rho,
    Kokkos::View<const double****> u,
    Kokkos::View<const double***> P,
    Kokkos::View<const double*> dx, 
    thermodynamics::PerfectGas const& eos);