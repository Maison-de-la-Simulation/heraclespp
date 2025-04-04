// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file user_step.hpp
//!

#pragma once

#include <kokkos_shortcut.hpp>

namespace novapp
{

class Range;

class IUserStep
{
public:
    IUserStep();

    IUserStep(IUserStep const& rhs);

    IUserStep(IUserStep&& rhs) noexcept;

    virtual ~IUserStep() noexcept;

    IUserStep& operator=(IUserStep const& rhs);

    IUserStep& operator=(IUserStep&& rhs) noexcept;

    virtual void execute(
        Range const &range,
        double t,
        double dt,
        KV_double_3d const& rho,
        KV_double_3d const& E,
        KV_double_4d const& fx) const = 0;
};

class NoUserStep : public IUserStep
{
public:
    void execute(
        Range const &range,
        double t,
        double dt,
        KV_double_3d const& rho,
        KV_double_3d const& E,
        KV_double_4d const& fx) const final;
};

class HeatNickelStep : public IUserStep
{
public:
    void execute(
        Range const &range,
        double t,
        double dt,
        KV_double_3d const& rho,
        KV_double_3d const& E,
        KV_double_4d const& fx) const final;
};

} // namespace novapp
