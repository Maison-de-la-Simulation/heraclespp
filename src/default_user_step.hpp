#pragma once

#include <stdexcept>

#include <user_step.hpp>

namespace novapp
{

class UserStep : public IUserStep
{
public:
    void execute(
        [[maybe_unused]] Range const &range,
        [[maybe_unused]] double const t,
        [[maybe_unused]] double const dt,
        [[maybe_unused]] KV_double_3d const& rho,
        [[maybe_unused]] KV_double_3d const& E,
        [[maybe_unused]] KV_double_4d const& fx) const final
    {
        throw std::runtime_error("User step not implemented");
    }
};

} // namespace novapp
