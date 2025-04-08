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
    explicit UserShiftCriterion([[maybe_unused]] ParamSetup const& param_setup)
    {
    }

    [[nodiscard]] bool execute(
            [[maybe_unused]] Range const& range,
            [[maybe_unused]] Grid const& grid,
            [[maybe_unused]] KV_double_3d const& rho,
            [[maybe_unused]] KV_double_4d const& rhou,
            [[maybe_unused]] KV_double_3d const& E,
            [[maybe_unused]] KV_double_4d const& fx) const final
    {
        throw std::runtime_error("User shift criterion not implemented");
    }
};

} // namespace novapp
