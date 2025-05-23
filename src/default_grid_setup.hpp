// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <stdexcept>

#include <grid_type.hpp>
#include <kokkos_shortcut.hpp>

namespace novapp
{

class Param;

class GridSetup : public IGridType
{
public:
    explicit GridSetup([[maybe_unused]] Param const& param)
    {
    }

    void execute(
        [[maybe_unused]] std::array<int, 3> Nghost,
        [[maybe_unused]] std::array<int, 3> Nx_glob_ng,
        [[maybe_unused]] KVH_double_1d const& x_glob,
        [[maybe_unused]] KVH_double_1d const& y_glob,
        [[maybe_unused]] KVH_double_1d const& z_glob) const final
    {
        throw std::runtime_error("Grid setup not implemented");
    }
};

} // namespace novapp
