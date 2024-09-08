//!
//! @file shift_criterion_interface.cpp
//!

#include <kokkos_shortcut.hpp>

#include "shift_criterion_interface.hpp"

namespace novapp
{

IShiftCriterion::IShiftCriterion() = default;

IShiftCriterion::IShiftCriterion(IShiftCriterion const& rhs) = default;

IShiftCriterion::IShiftCriterion(IShiftCriterion&& rhs) noexcept = default;

IShiftCriterion::~IShiftCriterion() noexcept = default;

IShiftCriterion& IShiftCriterion::operator=(IShiftCriterion const& rhs) = default;

IShiftCriterion& IShiftCriterion::operator=(IShiftCriterion&& rhs) noexcept = default;

bool NoShiftGrid::execute(
    [[maybe_unused]] Range const& range,
    [[maybe_unused]] Grid const& grid,
    [[maybe_unused]] KV_double_3d const& rho,
    [[maybe_unused]] KV_double_4d const& rhou,
    [[maybe_unused]] KV_double_3d const& E,
    [[maybe_unused]] KV_double_4d const& fx) const
{
    return false;
}

} // namespace novapp
