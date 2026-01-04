// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

//!
//! @file grid.hpp
//! Grid class declaration
//!

#pragma once

#include <mpi.h>

#include <array>
#include <iosfwd>

#include <kokkos_shortcut.hpp>

#include "range.hpp"

namespace hclpp {

class Grid
{
public:
    int Ng;
    std::array<int, 3> Nghost; // Number of ghost cells in each direction (default is 2)
    std::array<int, 3> Nx_glob_ng; // Total number of cells in each directions (excluding ghost)
    std::array<int, 3> Nx_local_ng; // Number of cells on the local MPI process (excluding ghost)
    std::array<int, 3> Nx_local_wg; // Number of cells on the local MPI process (including ghost)

    MPI_Comm comm_cart;
    MPI_Comm comm_cart_horizontal;
    int mpi_rank;
    int mpi_size;
    std::array<int, 3> mpi_rank_cart;
    std::array<int, 3> mpi_dims_cart;

    Range range;
    std::array<std::array<bool, 2>, 3> is_border;

    KV_double_1d x0;
    KV_double_1d x1;
    KV_double_1d x2;

    KV_double_1d x0_center;
    KV_double_1d x1_center;
    KV_double_1d x2_center;

    KV_double_1d dx0;
    KV_double_1d dx1;
    KV_double_1d dx2;

    KV_double_4d ds;
    KV_double_3d dv;

    Grid(std::array<int, 3> const& nx_glob_ng, std::array<int, 3> const& mpi_dims_cart, int Ng);

    Grid(Grid const& rhs) = delete;

    Grid(Grid&& rhs) noexcept = delete;

    ~Grid() noexcept;

    auto operator=(Grid const& rhs) -> Grid& = delete;

    auto operator=(Grid&& rhs) noexcept -> Grid& = delete;

    void print_grid(std::ostream& os) const;

    void set_grid(KV_cdouble_1d const& x0_glob, KV_cdouble_1d const& x1_glob, KV_cdouble_1d const& x2_glob);

private:
    void mpi_decomposition();
};

} // namespace hclpp
