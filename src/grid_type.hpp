//!
//! @file grid_type.hpp
//!

#pragma once

#include "grid.hpp"

namespace novapp
{

class IGridType
{
public:
    IGridType() = default;

    IGridType(IGridType const& x) = default;

    IGridType(IGridType&& x) noexcept = default;

    virtual ~IGridType() noexcept = default;

    IGridType& operator=(IGridType const& x) = default;

    IGridType& operator=(IGridType&& x) noexcept = default;

    virtual void execute(
        [[maybe_unused]] KVH_double_1d x_glob,
        [[maybe_unused]] KVH_double_1d y_glob,
        [[maybe_unused]] KVH_double_1d z_glob,
        [[maybe_unused]] std::array<int, 3> Nghost,
        [[maybe_unused]] std::array<int, 3> Nx_local_wg,
        [[maybe_unused]] std::array<int, 3> Nx_glob_ng) const 
        {
            throw std::runtime_error("Boundary not implemented");
        }
};

class Regular : public IGridType
{
private :
    Param m_param;

public:
    Regular(
        Param const& param)
        : IGridType()
        , m_param(param)
    {
    }

    void execute(
        KVH_double_1d x_glob,
        KVH_double_1d y_glob,
        KVH_double_1d z_glob,
        [[maybe_unused]] std::array<int, 3> Nghost,
        [[maybe_unused]] std::array<int, 3> Nx_local_wg,
        [[maybe_unused]] std::array<int, 3> Nx_glob_ng) const final
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
};

} // namespace novapp
