// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file geometry.hpp
//!

#pragma once

#include <kokkos_shortcut.hpp>

namespace hclpp {

class Range;

class IComputeGeom
{
public:
    IComputeGeom();

    IComputeGeom(IComputeGeom const& rhs);

    IComputeGeom(IComputeGeom&& rhs) noexcept;

    virtual ~IComputeGeom() noexcept;

    auto operator=(IComputeGeom const& rhs) -> IComputeGeom&;

    auto operator=(IComputeGeom&& rhs) noexcept -> IComputeGeom&;

    virtual void execute(
            Range const& range,
            KV_cdouble_1d const& x0,
            KV_cdouble_1d const& x1,
            KV_cdouble_1d const& x2,
            KV_cdouble_1d const& dx0,
            KV_cdouble_1d const& dx1,
            KV_cdouble_1d const& dx2,
            KV_double_4d const& ds,
            KV_double_3d const& dv) const
            = 0;
};

class Cartesian : public IComputeGeom
{
public:
    void execute(
            Range const& range,
            KV_cdouble_1d const& x0,
            KV_cdouble_1d const& x1,
            KV_cdouble_1d const& x2,
            KV_cdouble_1d const& dx0,
            KV_cdouble_1d const& dx1,
            KV_cdouble_1d const& dx2,
            KV_double_4d const& ds,
            KV_double_3d const& dv) const final;
};

class Spherical : public IComputeGeom
{
public:
    void execute(
            Range const& range,
            KV_cdouble_1d const& x0,
            KV_cdouble_1d const& x1,
            KV_cdouble_1d const& x2,
            KV_cdouble_1d const& dx0,
            KV_cdouble_1d const& dx1,
            KV_cdouble_1d const& dx2,
            KV_double_4d const& ds,
            KV_double_3d const& dv) const final;
};

} // namespace hclpp
