//!
//! @file initialization_interface.hpp
//!

#pragma once

#include <PerfectGas.hpp>

#include "range.hpp"
#include "kokkos_shortcut.hpp"
#include "grid.hpp"

namespace novapp
{

class ParamSetup;

class IInitializationProblem
{
public:
    IInitializationProblem() = default;

    IInitializationProblem(IInitializationProblem const& x) = default;

    IInitializationProblem(IInitializationProblem&& x) noexcept = default;

    virtual ~IInitializationProblem() noexcept = default;

    IInitializationProblem& operator=(IInitializationProblem const& x) = default;

    IInitializationProblem& operator=(IInitializationProblem&& x) noexcept = default;

    virtual void execute(
        Range const& range,
        KV_double_3d rho,
        KV_double_4d u,
        KV_double_3d P,
        KV_double_4d fx,
        KV_double_1d g) const
        = 0;
};

} // namespace novapp
