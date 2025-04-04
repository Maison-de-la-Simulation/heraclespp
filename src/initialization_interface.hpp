// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file initialization_interface.hpp
//!

#pragma once

#include <kokkos_shortcut.hpp>

namespace novapp
{

class Grid;
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
        Grid const& grid,
        KV_double_3d const& rho,
        KV_double_4d const& u,
        KV_double_3d const& P,
        KV_double_4d const& fx) const
        = 0;
};

} // namespace novapp
