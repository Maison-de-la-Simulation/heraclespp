// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

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
void compute_regular_mesh_1d(KVH_double_1d const& x, int const ng, double const xmin, double const dx) noexcept
{
    int const istart = 0;
    int const iend = x.extent_int(0);
    for (int i = istart; i < iend; ++i) {
        x(i) = xmin + ((i - ng) * dx);
    }
}

IGridType::IGridType() = default;

IGridType::IGridType(IGridType const& rhs) = default;

IGridType::IGridType(IGridType&& rhs) noexcept = default;

IGridType::~IGridType() noexcept = default;

IGridType& IGridType::operator=(IGridType const& /*rhs*/) = default;

IGridType& IGridType::operator=(IGridType&& /*rhs*/) noexcept = default;

Regular::Regular(std::array<double, 3> min, std::array<double, 3> max) : m_min(min), m_max(max) {}

Regular::Regular(Param const& param) : m_min {param.x0min, param.x1min, param.x2min}, m_max {param.x0max, param.x1max, param.x2max} {}

void Regular::execute(
        std::array<int, 3> Nghost,
        std::array<int, 3> Nx_glob_ng,
        KVH_double_1d const& x0_glob,
        KVH_double_1d const& x1_glob,
        KVH_double_1d const& x2_glob) const
{
    compute_regular_mesh_1d(x0_glob, Nghost[0], m_min[0], (m_max[0] - m_min[0]) / Nx_glob_ng[0]);
    compute_regular_mesh_1d(x1_glob, Nghost[1], m_min[1], (m_max[1] - m_min[1]) / Nx_glob_ng[1]);
    compute_regular_mesh_1d(x2_glob, Nghost[2], m_min[2], (m_max[2] - m_min[2]) / Nx_glob_ng[2]);
}

} // namespace novapp
