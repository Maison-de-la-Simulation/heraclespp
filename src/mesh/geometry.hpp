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
        KV_cdouble_1d const& x,
        KV_cdouble_1d const& y,
        KV_cdouble_1d const& z,
        KV_cdouble_1d const& dx,
        KV_cdouble_1d const& dy,
        KV_cdouble_1d const& dz,
        KV_double_4d const& ds,
        KV_double_3d const& dv) const
        = 0;
};

class Cartesian : public IComputeGeom
{
public:
    void execute(
        Range const& range,
        KV_cdouble_1d const& x,
        KV_cdouble_1d const& y,
        KV_cdouble_1d const& z,
        KV_cdouble_1d const& dx,
        KV_cdouble_1d const& dy,
        KV_cdouble_1d const& dz,
        KV_double_4d const& ds,
        KV_double_3d const& dv) const final;
};

class Spherical : public IComputeGeom
{
public:
    void execute(
        Range const& range,
        KV_cdouble_1d const& x,
        KV_cdouble_1d const& y,
        KV_cdouble_1d const& z,
        KV_cdouble_1d const& dx,
        KV_cdouble_1d const& dy,
        KV_cdouble_1d const& dz,
        KV_double_4d const& ds,
        KV_double_3d const& dv) const final;
};

} // namespace novapp
