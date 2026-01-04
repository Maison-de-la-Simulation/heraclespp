// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file shift_criterion_interface.hpp
//!

#pragma once

#include <kokkos_shortcut.hpp>

namespace hclpp {

class Grid;
class Range;

class IShiftCriterion
{
public:
    IShiftCriterion();

    IShiftCriterion(IShiftCriterion const& rhs);

    IShiftCriterion(IShiftCriterion&& rhs) noexcept;

    virtual ~IShiftCriterion() noexcept;

    auto operator=(IShiftCriterion const& rhs) -> IShiftCriterion&;

    auto operator=(IShiftCriterion&& rhs) noexcept -> IShiftCriterion&;

    [[nodiscard]] virtual auto execute(
            Range const& range,
            Grid const& grid,
            KV_cdouble_3d const& rho,
            KV_cdouble_4d const& rhou,
            KV_cdouble_3d const& E,
            KV_cdouble_4d const& fx) const -> bool
            = 0;
};

class NoShiftGrid : public IShiftCriterion
{
public:
    [[nodiscard]] auto execute(
            Range const& range,
            Grid const& grid,
            KV_cdouble_3d const& rho,
            KV_cdouble_4d const& rhou,
            KV_cdouble_3d const& E,
            KV_cdouble_4d const& fx) const -> bool final;
};

} // namespace hclpp
