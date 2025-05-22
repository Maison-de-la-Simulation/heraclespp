// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file initialization_interface.cpp
//!

#include "initialization_interface.hpp"

namespace novapp {

IInitializationProblem::IInitializationProblem() = default;

IInitializationProblem::IInitializationProblem(IInitializationProblem const& rhs) = default;

IInitializationProblem::IInitializationProblem(IInitializationProblem&& rhs) noexcept = default;

IInitializationProblem::~IInitializationProblem() noexcept = default;

IInitializationProblem& IInitializationProblem::operator=(IInitializationProblem const& /*rhs*/) = default;

IInitializationProblem& IInitializationProblem::operator=(IInitializationProblem&& /*rhs*/) noexcept = default;

} // namespace novapp
