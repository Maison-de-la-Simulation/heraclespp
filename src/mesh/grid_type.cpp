//!
//! @file grid_type.cpp
//!

#include <array>
#include <stdexcept>

#include <kokkos_shortcut.hpp>
#include <nova_params.hpp>

#include "grid_type.hpp"

namespace novapp {

IGridType::IGridType() = default;

IGridType::IGridType([[maybe_unused]] IGridType const& rhs) = default;

IGridType::IGridType([[maybe_unused]] IGridType&& rhs) noexcept = default;

IGridType::~IGridType() noexcept = default;

IGridType& IGridType::operator=([[maybe_unused]] IGridType const& rhs) = default;

IGridType& IGridType::operator=([[maybe_unused]] IGridType&& rhs) noexcept = default;

void IGridType::execute(
        [[maybe_unused]] KVH_double_1d x_glob,
        [[maybe_unused]] KVH_double_1d y_glob,
        [[maybe_unused]] KVH_double_1d z_glob,
        [[maybe_unused]] std::array<int, 3> Nghost,
        [[maybe_unused]] std::array<int, 3> Nx_glob_ng) const
{
    throw std::runtime_error("Boundary not implemented");
}

Regular::Regular(Param const& param)
   : m_param(param)
{
}

void Regular::execute(
    KVH_double_1d x_glob,
    KVH_double_1d y_glob,
    KVH_double_1d z_glob,
    std::array<int, 3> Nghost,
    std::array<int, 3> Nx_glob_ng) const
{
    double Lx = m_param.xmax - m_param.xmin;
    double Ly = m_param.ymax - m_param.ymin;
    double Lz = m_param.zmax - m_param.zmin;

    double dx = Lx / Nx_glob_ng[0];
    double dy = Ly / Nx_glob_ng[1];
    double dz = Lz / Nx_glob_ng[2];

    x_glob(Nghost[0]) = m_param.xmin;
    y_glob(Nghost[1]) = m_param.ymin;
    z_glob(Nghost[2]) = m_param.zmin;

    for (int i = Nghost[0]+1; i < x_glob.extent_int(0) ; i++)
    {
        x_glob(i) = x_glob(i-1) + dx;
    }
    for (int i = Nghost[1]+1; i < y_glob.extent_int(0) ; i++)
    {
        y_glob(i) = y_glob(i-1) + dy;
    }
    for (int i = Nghost[2]+1; i < z_glob.extent_int(0) ; i++)
    {
        z_glob(i) = z_glob(i-1) + dz;
    }

    // Left ghost cells
    for(int i = Nghost[0]-1; i >= 0; i--)
    {
        x_glob(i) = x_glob(i+1) - dx;
    }
    for(int i = Nghost[1]-1; i >= 0; i--)
    {
        y_glob(i) = y_glob(i+1) - dy;
    }
    for(int i = Nghost[2]-1; i >= 0; i--)
    {
        z_glob(i) = z_glob(i+1) - dz;
    }
}

} // namespace novapp
