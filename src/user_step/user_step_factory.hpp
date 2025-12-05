// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file user_step_factory.hpp
//!

#pragma once

#include <memory>
#include <string>

namespace hclpp {

class IUserStep;

std::unique_ptr<IUserStep> factory_user_step(std::string const& user_step);

} // namespace hclpp
