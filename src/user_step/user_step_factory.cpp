//!
//! @file user_step_factory.cpp
//!

#include <memory>
#include <stdexcept>
#include <string>

#include "user_step.hpp"
#include "user_step_factory.hpp"

namespace novapp
{

std::unique_ptr<IUserStep> factory_user_step(std::string const& user_step)
{
    if (user_step == "Off")
    {
        return std::make_unique<NoUserStep>();
    }

    if (user_step == "Heat_nickel")
    {
        return std::make_unique<HeatNickelStep>();
    }

    throw std::runtime_error("Unknown user step: " + user_step + ".");
}

} // namespace novapp
