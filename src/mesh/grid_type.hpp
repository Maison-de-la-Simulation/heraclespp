//!
//! @file grid_type.hpp
//!

#pragma once

#include <array>

#include <kokkos_shortcut.hpp>
#include <nova_params.hpp>

namespace novapp
{

class IGridType
{
public:
    IGridType();

    IGridType(IGridType const& rhs);

    IGridType(IGridType&& rhs) noexcept;

    virtual ~IGridType() noexcept;

    IGridType& operator=(IGridType const& rhs);

    IGridType& operator=(IGridType&& rhs) noexcept;

    virtual void execute(
        std::array<int, 3> Nghost,
        std::array<int, 3> Nx_glob_ng,
        KVH_double_1d x_glob,
        KVH_double_1d y_glob,
        KVH_double_1d z_glob) const = 0;
};

class Regular : public IGridType
{
private :
    Param m_param;

public:
    explicit Regular(Param const& param);

    void execute(
        std::array<int, 3> Nghost,
        std::array<int, 3> Nx_glob_ng,
        KVH_double_1d x_glob,
        KVH_double_1d y_glob,
        KVH_double_1d z_glob) const final;
};

} // namespace novapp
