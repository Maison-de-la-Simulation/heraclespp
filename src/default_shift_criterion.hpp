// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>

#include <kokkos_shortcut.hpp>
#include <shift_criterion_interface.hpp>

namespace novapp {

class Grid;
class ParamSetup;
class Range;

class UserShiftCriterion : public IShiftCriterion
{
public:
    explicit UserShiftCriterion(ParamSetup const& /*param_setup*/)
    {
    }

    [[nodiscard]] bool execute(
            Range const& /*range*/,
            Grid const& /*grid*/,
            KV_cdouble_3d const& /*rho*/,
            KV_cdouble_4d const& /*rhou*/,
            KV_cdouble_3d const& /*E*/,
            KV_cdouble_4d const& /*fx*/) const final
    {
        throw std::runtime_error("User shift criterion not implemented");
    }
};

} // namespace novapp
