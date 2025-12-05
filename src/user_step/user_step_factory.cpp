// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file user_step_factory.cpp
//!

#include <memory>
#include <stdexcept>
#include <string>

#include "user_step.hpp"
#include "user_step_factory.hpp"

namespace hclpp {

std::unique_ptr<IUserStep> factory_user_step(std::string const& user_step)
{
    if (user_step == "Off") {
        return std::make_unique<NoUserStep>();
    }

    if (user_step == "Heat_nickel") {
        return std::make_unique<HeatNickelStep>();
    }

    throw std::runtime_error("Unknown user step: " + user_step + ".");
}

} // namespace hclpp
