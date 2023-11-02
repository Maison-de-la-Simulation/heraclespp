//!
//! @file initialization_interface.hpp
//!

#pragma once

#include <kokkos_shortcut.hpp>

namespace novapp
{

class Range;

class IInitializationProblem
{
public:
    IInitializationProblem();

    IInitializationProblem(IInitializationProblem const& rhs);

    IInitializationProblem(IInitializationProblem&& rhs) noexcept;

    virtual ~IInitializationProblem() noexcept;

    IInitializationProblem& operator=(IInitializationProblem const& rhs);

    IInitializationProblem& operator=(IInitializationProblem&& rhs) noexcept;

    virtual void execute(
        Range const& range,
        KV_double_3d rho,
        KV_double_4d u,
        KV_double_3d P,
        KV_double_4d fx) const
        = 0;
};

} // namespace novapp
