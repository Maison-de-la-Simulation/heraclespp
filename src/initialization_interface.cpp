// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file initialization_interface.cpp
//!

#include "initialization_interface.hpp"

namespace hclpp {

IInitializationProblem::IInitializationProblem() = default;

IInitializationProblem::IInitializationProblem(IInitializationProblem const& rhs) = default;

IInitializationProblem::IInitializationProblem(IInitializationProblem&& rhs) noexcept = default;

IInitializationProblem::~IInitializationProblem() noexcept = default;

auto IInitializationProblem::operator=(IInitializationProblem const& /*rhs*/) -> IInitializationProblem& = default;

auto IInitializationProblem::operator=(IInitializationProblem&& /*rhs*/) noexcept -> IInitializationProblem& = default;

} // namespace hclpp
