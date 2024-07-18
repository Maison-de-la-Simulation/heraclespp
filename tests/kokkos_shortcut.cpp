#include <utility>

#include <gtest/gtest.h>

#include <kokkos_shortcut.hpp>

TEST(EqualExtents, SingleDimension)
{
    Kokkos::DualView<double****> const v0("v0", 2, 3, 4, 5);
    Kokkos::View<double***> const v1("v1", 2, 0, 4);
    EXPECT_TRUE(novapp::equal_extents(0, v0, v1));
    EXPECT_FALSE(novapp::equal_extents(1, v0, v1));
    EXPECT_TRUE(novapp::equal_extents(2, v0, v1));
}

TEST(EqualExtents, MultipleDimensions)
{
    Kokkos::DualView<double****> const v0("v0", 2, 3, 4, 5);
    Kokkos::View<double***> const v1("v1", 2, 0, 4);
    EXPECT_TRUE(novapp::equal_extents({0, 2}, v0, v1));
    EXPECT_FALSE(novapp::equal_extents({1}, v0, v1));
}
