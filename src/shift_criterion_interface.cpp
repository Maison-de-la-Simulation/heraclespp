// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file shift_criterion_interface.cpp
//!

#include <cassert>

#include <kokkos_shortcut.hpp>
#if !defined(NDEBUG)
#    include <ndim.hpp>
#endif

#include "shift_criterion_interface.hpp"

namespace novapp
{

IShiftCriterion::IShiftCriterion() = default;

IShiftCriterion::IShiftCriterion(IShiftCriterion const& rhs) = default;

IShiftCriterion::IShiftCriterion(IShiftCriterion&& rhs) noexcept = default;

IShiftCriterion::~IShiftCriterion() noexcept = default;

IShiftCriterion& IShiftCriterion::operator=(IShiftCriterion const& /*rhs*/) = default;

IShiftCriterion& IShiftCriterion::operator=(IShiftCriterion&& /*rhs*/) noexcept = default;

bool NoShiftGrid::execute(
    Range const& /*range*/,
    Grid const& /*grid*/,
    [[maybe_unused]] KV_cdouble_3d const& rho,
    [[maybe_unused]] KV_cdouble_4d const& rhou,
    [[maybe_unused]] KV_cdouble_3d const& E,
    [[maybe_unused]] KV_cdouble_4d const& fx) const
{
    assert(equal_extents({0, 1, 2}, rho, rhou, E, fx));
    assert(rhou.extent_int(3) == ndim);

    return false;
}

} // namespace novapp
