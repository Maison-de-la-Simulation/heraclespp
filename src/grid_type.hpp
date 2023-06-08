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
        KVH_double_1d x_glob,
        KVH_double_1d y_glob,
        KVH_double_1d z_glob,
        std::array<int, 3> Nghost,
        std::array<int, 3> Nx_local_wg,
        std::array<int, 3> Nx_glob_ng) const 
        {
            throw std::runtime_error("Boundary not implemented");
        }
};

class Regular : public IGridType
{
public:
    Regular()
    : IGridType()
    {
    }

    void execute(
        [[maybe_unused]] KVH_double_1d x_glob,
        [[maybe_unused]] KVH_double_1d y_glob,
        [[maybe_unused]] KVH_double_1d z_glob,
        [[maybe_unused]] std::array<int, 3> Nghost,
        [[maybe_unused]] std::array<int, 3> Nx_local_wg,
        [[maybe_unused]] std::array<int, 3> Nx_glob_ng) const final
    {
        // do nothing
    }
};

} // namespace novapp
