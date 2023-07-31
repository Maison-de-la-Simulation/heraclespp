//!
//! @file initialization_interface.hpp
//!

#pragma once

#include "kokkos_shortcut.hpp"
#include "range.hpp"

namespace novapp
{

class IInitializationProblem
{
public:
    IInitializationProblem() = default;

    IInitializationProblem(IInitializationProblem const& rhs) = default;

    IInitializationProblem(IInitializationProblem&& rhs) noexcept = default;

    virtual ~IInitializationProblem() noexcept = default;

    IInitializationProblem& operator=(IInitializationProblem const& rhs) = default;

    IInitializationProblem& operator=(IInitializationProblem&& rhs) noexcept = default;

    virtual void execute(
        Range const& range,
        KV_double_3d rho,
        KV_double_4d u,
        KV_double_3d P,
        KV_double_4d fx) const
        = 0;
};

} // namespace novapp
