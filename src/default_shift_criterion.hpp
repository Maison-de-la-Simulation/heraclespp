#pragma once

#include <grid.hpp>
#include "range.hpp"
#include <shift_criterion_interface.hpp>

namespace novapp {

class ParamSetup;

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
