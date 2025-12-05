// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <string>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <kokkos_shortcut.hpp>

TEST(EqualExtents, SingleDimension)
{
    Kokkos::DualView<double****> const v0("v0", 2, 3, 4, 5);
    Kokkos::View<double***> const v1("v1", 2, 0, 4);
    EXPECT_TRUE(hclpp::equal_extents(0, v0, v1));
    EXPECT_FALSE(hclpp::equal_extents(1, v0, v1));
    EXPECT_TRUE(hclpp::equal_extents(2, v0, v1));
}

TEST(EqualExtents, MultipleDimensions)
{
    Kokkos::DualView<double****> const v0("v0", 2, 3, 4, 5);
    Kokkos::View<double***> const v1("v1", 2, 0, 4);
    EXPECT_TRUE(hclpp::equal_extents({0, 2}, v0, v1));
    EXPECT_FALSE(hclpp::equal_extents({1}, v0, v1));
}
