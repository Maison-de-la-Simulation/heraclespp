#include <gtest/gtest.h>

#include <gravity.hpp>
#include <grid_type.hpp>
#include <units.hpp>

#include "kokkos_shortcut.hpp"

namespace {

void TestGravityUniformGravity()
{
    double const gx = 1.4;
    double const gy = 1.9;
    double const gz = 3.7;
    novapp::KDV_double_1d g("g", 3);
    g.h_view(0) = gx;
    g.h_view(1) = gy;
    g.h_view(2) = gz;
    g.modify_host();
    g.sync_device();
    novapp::UniformGravity const gravity(g.d_view);
    int const nx = 5;
    int const ny = 4;
    int const nz = 3;
    novapp::KDV_double_4d result("result", nx, ny, nz, 3);
    Kokkos::parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<3>, int>({0, 0, 0}, {nx, ny, nz}),
            KOKKOS_LAMBDA(int i, int j, int k) {
                result.d_view(i, j, k, 0) = gravity(i, j, k, 0);
                result.d_view(i, j, k, 1) = gravity(i, j, k, 1);
                result.d_view(i, j, k, 2) = gravity(i, j, k, 2);
            });
    result.modify_device();
    result.sync_host();
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                EXPECT_EQ(result.h_view(i, j, k, 0), gx);
                EXPECT_EQ(result.h_view(i, j, k, 1), gy);
                EXPECT_EQ(result.h_view(i, j, k, 2), gz);
            }
        }
    }
}

} // namespace

TEST(Gravity, UniformGravity)
{
    TestGravityUniformGravity();
}
