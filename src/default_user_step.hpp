// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>

#include <kokkos_shortcut.hpp>
#include <user_step.hpp>

namespace hclpp {

class Range;

class UserStep : public IUserStep
{
public:
    void execute(
            Range const& /*range*/,
            double const /*t*/,
            double const /*dt*/,
            KV_double_3d const& /*rho*/,
            KV_double_3d const& /*E*/,
            KV_double_4d const& /*fx*/) const final
    {
        throw std::runtime_error("User step not implemented");
    }
};

} // namespace hclpp
