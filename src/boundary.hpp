// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file boundary.hpp
//!

#pragma once

#include <string>
#include <string_view>

#include <concepts.hpp>
#include <kokkos_shortcut.hpp>
#include <ndim.hpp>

namespace novapp
{

class Grid;

std::string_view bc_dir(int i) noexcept;

std::string_view bc_face(int i) noexcept;

void null_gradient_condition(int m_bc_idim, int m_bc_iface,
                             std::string const& m_label,
                             Grid const& grid,
                             KV_double_3d const& rho,
                             KV_double_4d const& rhou,
                             KV_double_3d const& E,
                             KV_double_4d const& fx);

void reflexive_condition(int m_bc_idim, int m_bc_iface,
                         std::string const& m_label,
                         Grid const& grid,
                         KV_double_3d const& rho,
                         KV_double_4d const& rhou,
                         KV_double_3d const& E,
                         KV_double_4d const& fx);

template <concepts::GravityField Gravity>
class IBoundaryCondition
{
private:
    int m_bc_idim;

    int m_bc_iface;

protected:
    [[nodiscard]] int bc_idim() const noexcept
    {
        return m_bc_idim;
    }

    [[nodiscard]] int bc_iface() const noexcept
    {
        return m_bc_iface;
    }

public:
    IBoundaryCondition(int idim, int iface) : m_bc_idim(idim), m_bc_iface(iface) {}

    IBoundaryCondition(IBoundaryCondition const& rhs) = default;

    IBoundaryCondition(IBoundaryCondition&& rhs) noexcept = default;

    virtual ~IBoundaryCondition() noexcept = default;

    IBoundaryCondition& operator=(IBoundaryCondition const& rhs) = default;

    IBoundaryCondition& operator=(IBoundaryCondition&& rhs) noexcept = default;

    virtual void execute(Grid const& grid,
                         Gravity const& gravity,
                         KV_double_3d const& rho,
                         KV_double_4d const& rhou,
                         KV_double_3d const& E,
                         KV_double_4d const& fx) const = 0;
};

template <concepts::GravityField Gravity>
class NullGradient : public IBoundaryCondition<Gravity>
{
private:
    std::string m_label;

public:
    NullGradient(int idim, int iface)
        : IBoundaryCondition<Gravity>(idim, iface)
        , m_label(std::string("NullGradient").append(bc_dir(idim)).append(bc_face(iface)))
    {
    }

    void execute(Grid const& grid,
                 [[maybe_unused]] Gravity const& gravity,
                 KV_double_3d const& rho,
                 KV_double_4d const& rhou,
                 KV_double_3d const& E,
                 KV_double_4d const& fx) const final
    {
        assert(equal_extents({0, 1, 2}, rho, rhou, E, fx));
        assert(rhou.extent_int(3) == ndim);

        null_gradient_condition(this->bc_idim(), this->bc_iface(), m_label, grid, rho, rhou, E, fx);
    }
};

template <concepts::GravityField Gravity>
class PeriodicCondition : public IBoundaryCondition<Gravity>
{
public:
    PeriodicCondition(int idim, int iface) : IBoundaryCondition<Gravity>(idim, iface) {}

    void execute(
            [[maybe_unused]] Grid const& grid,
            [[maybe_unused]] Gravity const& gravity,
            [[maybe_unused]] KV_double_3d const& rho,
            [[maybe_unused]] KV_double_4d const& rhou,
            [[maybe_unused]] KV_double_3d const& E,
            [[maybe_unused]] KV_double_4d const& fx) const final
    {
        assert(equal_extents({0, 1, 2}, rho, rhou, E, fx));
        assert(rhou.extent_int(3) == ndim);

        // do nothing
    }
};

template <concepts::GravityField Gravity>
class ReflexiveCondition : public IBoundaryCondition<Gravity>
{
private:
    std::string m_label;

public:
    ReflexiveCondition(int idim, int iface)
        : IBoundaryCondition<Gravity>(idim, iface)
        , m_label(std::string("Reflexive").append(bc_dir(idim)).append(bc_face(iface)))
    {
    }

    void execute(Grid const& grid,
                 [[maybe_unused]] Gravity const& gravity,
                 KV_double_3d const& rho,
                 KV_double_4d const& rhou,
                 KV_double_3d const& E,
                 KV_double_4d const& fx) const final
    {
        assert(equal_extents({0, 1, 2}, rho, rhou, E, fx));
        assert(rhou.extent_int(3) == ndim);

        reflexive_condition(this->bc_idim(), this->bc_iface(), m_label, grid, rho, rhou, E, fx);
    }
};

} // namespace novapp
