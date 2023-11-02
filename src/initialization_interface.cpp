//!
//! @file initialization_interface.cpp
//!

#include "initialization_interface.hpp"

namespace novapp
{

IInitializationProblem::IInitializationProblem() = default;

IInitializationProblem::IInitializationProblem([[maybe_unused]] IInitializationProblem const& rhs) = default;

IInitializationProblem::IInitializationProblem([[maybe_unused]] IInitializationProblem&& rhs) noexcept = default;

IInitializationProblem::~IInitializationProblem() noexcept = default;

IInitializationProblem& IInitializationProblem::operator=([[maybe_unused]] IInitializationProblem const& rhs) = default;

IInitializationProblem& IInitializationProblem::operator=([[maybe_unused]] IInitializationProblem&& rhs) noexcept = default;

} // namespace novapp
