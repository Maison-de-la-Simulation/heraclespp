// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>

namespace hclpp {

struct EulerCons;
struct EulerFlux;

namespace concepts {

template <class Solver, class EoS>
concept EulerRiemannSolver = requires(Solver const solver, EulerCons const& s, int const i, EoS const eos) {
    { solver(s, s, i, eos) } noexcept -> std::same_as<EulerFlux>;
};

template <class Limiter>
concept SlopeLimiter = requires(Limiter const limiter, double const s) {
    { limiter(s, s) } noexcept -> std::same_as<double>;
};

template <class EoS>
concept EulerEoS = requires(EoS const eos, double const x) {
    { eos.compute_evol_from_pres(x, x) } noexcept -> std::same_as<double>;
    { eos.compute_pres_from_evol(x, x) } noexcept -> std::same_as<double>;
    { eos.compute_speed_of_sound(x, x) } noexcept -> std::same_as<double>;
};

template <class Gravity>
concept GravityField = requires(Gravity const gravity, int const i) {
    { gravity(i, i, i, i) } noexcept -> std::same_as<double>;
};

} // namespace concepts

} // namespace hclpp
