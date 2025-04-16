// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file boundary.hpp
//!

#pragma once

#include <string>
#include <string_view>

#include <kokkos_shortcut.hpp>

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

template <class Gravity>
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

template <class Gravity>
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
        null_gradient_condition(this->bc_idim(), this->bc_iface(), m_label, grid, rho, rhou, E, fx);
    }
};

template <class Gravity>
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
        // do nothing
    }
};

template <class Gravity>
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
        reflexive_condition(this->bc_idim(), this->bc_iface(), m_label, grid, rho, rhou, E, fx);
    }
};

} // namespace novapp
