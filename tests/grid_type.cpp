#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <grid_type.hpp>

TEST(GridType, ComputeRegularMesh1d)
{
    int const hw = 1;
    Kokkos::View<double*, Kokkos::HostSpace> const x("x", 3 + 2 * hw);
    novapp::compute_regular_mesh_1d(x, hw, -1., 1.);
    EXPECT_DOUBLE_EQ(x(0), -2.);
    EXPECT_DOUBLE_EQ(x(1), -1.);
    EXPECT_DOUBLE_EQ(x(2), 0.);
    EXPECT_DOUBLE_EQ(x(3), +1.);
    EXPECT_DOUBLE_EQ(x(4), +2.);
}
