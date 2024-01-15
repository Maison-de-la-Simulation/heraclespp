//!
//! @file geometry.hpp
//!

#pragma once

#include <kokkos_shortcut.hpp>

namespace novapp
{

class Range;

class IComputeGeom
{
public:
    IComputeGeom();

    IComputeGeom(IComputeGeom const& rhs);

    IComputeGeom(IComputeGeom&& rhs) noexcept;

    virtual ~IComputeGeom() noexcept;

    IComputeGeom& operator=(IComputeGeom const& rhs);

    IComputeGeom& operator=(IComputeGeom&& rhs) noexcept;

    virtual void execute(
        Range const& range,
        KV_cdouble_1d x,
        KV_cdouble_1d y,
        KV_cdouble_1d z,
        KV_cdouble_1d dx,
        KV_cdouble_1d dy,
        KV_cdouble_1d dz,
        KV_double_4d ds,
        KV_double_3d dv) const
        = 0;
};

class Cartesian : public IComputeGeom
{
public:
    void execute(
        Range const& range,
        KV_cdouble_1d x,
        KV_cdouble_1d y,
        KV_cdouble_1d z,
        KV_cdouble_1d dx,
        KV_cdouble_1d dy,
        KV_cdouble_1d dz,
        KV_double_4d ds,
        KV_double_3d dv) const final;
};

class Spherical : public IComputeGeom
{
public:
    void execute(
        Range const& range,
        KV_cdouble_1d x,
        KV_cdouble_1d y,
        KV_cdouble_1d z,
        KV_cdouble_1d dx,
        KV_cdouble_1d dy,
        KV_cdouble_1d dz,
        KV_double_4d ds,
        KV_double_3d dv) const final;
};

} // namespace novapp
