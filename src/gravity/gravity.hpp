// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file gravity.hpp
//!

#pragma once

#include <Kokkos_Core.hpp>
#include <kokkos_shortcut.hpp>

namespace novapp
{

class Grid;

class UniformGravity
{
private :
    KV_cdouble_1d m_g;

public :
    explicit UniformGravity(KV_cdouble_1d g);

    UniformGravity(const UniformGravity& rhs) = default;

    UniformGravity(UniformGravity&& rhs) noexcept = default;

    ~UniformGravity() noexcept = default;

    UniformGravity& operator=(const UniformGravity& rhs) = default;

    UniformGravity& operator=(UniformGravity&& rhs) noexcept = default;

    KOKKOS_FORCEINLINE_FUNCTION
    double operator()(int /*i*/, int /*j*/, int /*k*/, int dir) const noexcept
    {
        return m_g(dir);
    }
};

UniformGravity make_uniform_gravity(double gx0, double gx1, double gx2);

class PointMassGravity
{
private :
    KV_cdouble_1d m_g;

public :
    explicit PointMassGravity(KV_cdouble_1d g);

    PointMassGravity(const PointMassGravity& rhs) = default;

    PointMassGravity(PointMassGravity&& rhs) noexcept = default;

    ~PointMassGravity() noexcept = default;

    PointMassGravity& operator=(const PointMassGravity& rhs) = default;

    PointMassGravity& operator=(PointMassGravity&& rhs) noexcept = default;

    KOKKOS_FORCEINLINE_FUNCTION
    double operator()(int i, int /*j*/, int /*k*/, int dir) const noexcept
    {
        if (dir == 0)
        {
            return m_g(i);
        }
        return 0;
    }
};

PointMassGravity make_point_mass_gravity(
    double central_mass,
    Grid const& grid);

class InternalMassGravity
{
private :
    KV_cdouble_1d m_g;

public :
    explicit InternalMassGravity(KV_cdouble_1d g);

    InternalMassGravity(const InternalMassGravity& rhs) = default;

    InternalMassGravity(InternalMassGravity&& rhs) noexcept = default;

    ~InternalMassGravity() noexcept = default;

    InternalMassGravity& operator=(const InternalMassGravity& rhs) = default;

    InternalMassGravity& operator=(InternalMassGravity&& rhs) noexcept = default;

    KOKKOS_FORCEINLINE_FUNCTION
    double operator()(int i, int /*j*/, int /*k*/, int dir) const noexcept
    {
        if (dir == 0)
        {
            return m_g(i);
        }
        return 0;
    }
};

InternalMassGravity make_internal_mass_gravity(
    double central_mass,
    Grid const& grid,
    KV_cdouble_3d const& rho);

} // namespace novapp
