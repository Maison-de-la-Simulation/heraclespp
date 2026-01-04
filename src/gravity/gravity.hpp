// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file gravity.hpp
//!

#pragma once

#include <Kokkos_Core.hpp>
#include <kokkos_shortcut.hpp>

namespace hclpp {

class Grid;

class UniformGravity
{
private:
    KV_cdouble_1d m_g;

public:
    explicit UniformGravity(KV_cdouble_1d g);

    UniformGravity(UniformGravity const& rhs) = default;

    UniformGravity(UniformGravity&& rhs) noexcept = default;

    ~UniformGravity() noexcept = default;

    auto operator=(UniformGravity const& rhs) -> UniformGravity& = default;

    auto operator=(UniformGravity&& rhs) noexcept -> UniformGravity& = default;

    KOKKOS_FORCEINLINE_FUNCTION
    auto operator()(int /*i*/, int /*j*/, int /*k*/, int dir) const noexcept -> double
    {
        return m_g(dir);
    }
};

auto make_uniform_gravity(double gx0, double gx1, double gx2) -> UniformGravity;

class PointMassGravity
{
private:
    KV_cdouble_1d m_g;

public:
    explicit PointMassGravity(KV_cdouble_1d g);

    PointMassGravity(PointMassGravity const& rhs) = default;

    PointMassGravity(PointMassGravity&& rhs) noexcept = default;

    ~PointMassGravity() noexcept = default;

    auto operator=(PointMassGravity const& rhs) -> PointMassGravity& = default;

    auto operator=(PointMassGravity&& rhs) noexcept -> PointMassGravity& = default;

    KOKKOS_FORCEINLINE_FUNCTION
    auto operator()(int i, int /*j*/, int /*k*/, int dir) const noexcept -> double
    {
        if (dir == 0) {
            return m_g(i);
        }
        return 0;
    }
};

auto make_point_mass_gravity(double central_mass, Grid const& grid) -> PointMassGravity;

class InternalMassGravity
{
private:
    KV_cdouble_1d m_g;

public:
    explicit InternalMassGravity(KV_cdouble_1d g);

    InternalMassGravity(InternalMassGravity const& rhs) = default;

    InternalMassGravity(InternalMassGravity&& rhs) noexcept = default;

    ~InternalMassGravity() noexcept = default;

    auto operator=(InternalMassGravity const& rhs) -> InternalMassGravity& = default;

    auto operator=(InternalMassGravity&& rhs) noexcept -> InternalMassGravity& = default;

    KOKKOS_FORCEINLINE_FUNCTION
    auto operator()(int i, int /*j*/, int /*k*/, int dir) const noexcept -> double
    {
        if (dir == 0) {
            return m_g(i);
        }
        return 0;
    }
};

auto make_internal_mass_gravity(double central_mass, Grid const& grid, KV_cdouble_3d const& rho) -> InternalMassGravity;

} // namespace hclpp
