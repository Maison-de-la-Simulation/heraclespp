//!
//! @file grid_type.cpp
//!

#include <array>

#include <kokkos_shortcut.hpp>
#include <nova_params.hpp>

#include "grid_type.hpp"

namespace novapp {

//! Computes the position of nodes in a regular mesh given the length of cells
//! @param[out] x The coordinates of nodes
//! @param[in] ng The number of halo cells
//! @param[in] xmin The coordinate of xmin
//! @param[in] dx The length of cells
void compute_regular_mesh_1d(
        KVH_double_1d const& x,
        int const ng,
        double const xmin,
        double const dx) noexcept
{
    int const istart = 0;
    int const iend = x.extent_int(0);
    for (int i = istart; i < iend; ++i)
    {
        x(i) = xmin + (i - ng) * dx;
    }
}

IGridType::IGridType() = default;

IGridType::IGridType(IGridType const& rhs) = default;

IGridType::IGridType(IGridType&& rhs) noexcept = default;

IGridType::~IGridType() noexcept = default;

IGridType& IGridType::operator=(IGridType const& rhs) = default;

IGridType& IGridType::operator=(IGridType&& rhs) noexcept = default;


Regular::Regular(Param const& param)
   : m_param(param)
{
}

void Regular::execute(
    std::array<int, 3> Nghost,
    std::array<int, 3> Nx_glob_ng,
    KVH_double_1d x_glob,
    KVH_double_1d y_glob,
    KVH_double_1d z_glob) const
{
    compute_regular_mesh_1d(x_glob, Nghost[0], m_param.xmin, (m_param.xmax - m_param.xmin) / Nx_glob_ng[0]);
    compute_regular_mesh_1d(y_glob, Nghost[1], m_param.ymin, (m_param.ymax - m_param.ymin) / Nx_glob_ng[1]);
    compute_regular_mesh_1d(z_glob, Nghost[2], m_param.zmin, (m_param.zmax - m_param.zmin) / Nx_glob_ng[2]);
}

} // namespace novapp
