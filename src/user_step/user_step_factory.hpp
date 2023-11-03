//!
//! @file user_step_factory.hpp
//!

#pragma once

#include <memory>
#include <string>

namespace novapp
{

class IUserStep;

std::unique_ptr<IUserStep> factory_user_step(std::string const& user_step);

} // namespace novapp
