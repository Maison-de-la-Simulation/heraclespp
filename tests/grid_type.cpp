// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <grid_type.hpp>
#include <kokkos_shortcut.hpp>

TEST(GridType, ComputeRegularMesh1d)
{
    int const hw = 1;
    novapp::KVH_double_1d const x("x", 3 + (2 * hw));
    novapp::compute_regular_mesh_1d(x, hw, -1., 1.);
    EXPECT_DOUBLE_EQ(x(0), -2.);
    EXPECT_DOUBLE_EQ(x(1), -1.);
    EXPECT_DOUBLE_EQ(x(2), 0.);
    EXPECT_DOUBLE_EQ(x(3), +1.);
    EXPECT_DOUBLE_EQ(x(4), +2.);
}
