//!
//! @file boundary.hpp
//!

#pragma once

#include <array>
#include <string>

#include <grid.hpp>
#include <kokkos_shortcut.hpp>
#include <ndim.hpp>

namespace novapp
{

extern std::array<std::string, 3> const bc_dir;
extern std::array<std::string, 2> const bc_face;

class IBoundaryCondition
{
protected:
    int m_bc_idim;
    int m_bc_iface;

public:
    IBoundaryCondition(int idim, int iface);

    IBoundaryCondition(IBoundaryCondition const& rhs);

    IBoundaryCondition(IBoundaryCondition&& rhs) noexcept;

    virtual ~IBoundaryCondition() noexcept;

    IBoundaryCondition& operator=(IBoundaryCondition const& rhs);

    IBoundaryCondition& operator=(IBoundaryCondition&& rhs) noexcept;

    virtual void execute(KV_double_3d rho,
                         KV_double_4d rhou,
                         KV_double_3d E,
                         KV_double_4d fx) const = 0;
};

class NullGradient : public IBoundaryCondition
{
private:
    std::string m_label;
    Grid m_grid;

public:
    NullGradient(int idim, int iface, Grid const& grid);

    void execute(KV_double_3d rho,
                 KV_double_4d rhou,
                 KV_double_3d E,
                 KV_double_4d fx) const final;
};

class PeriodicCondition : public IBoundaryCondition
{
public:
    PeriodicCondition(int idim, int iface);

    void execute(KV_double_3d rho,
                 KV_double_4d rhou,
                 KV_double_3d E,
                 KV_double_4d fx) const final;
};

class ReflexiveCondition : public IBoundaryCondition
{
private:
    std::string m_label;
    Grid m_grid;

public:
    ReflexiveCondition(int idim, int iface, Grid const& grid);

    void execute(KV_double_3d rho,
                 KV_double_4d rhou,
                 KV_double_3d E,
                 KV_double_4d fx) const final;
};

} // namespace novapp
