#include <gtest/gtest.h>

#include <broadcast.hpp>
#include <ndim.hpp>
#include <range.hpp>

TEST(BroadcastScalar, Case1d)
{
    if (novapp::ndim != 1) {
        GTEST_SKIP();
    }
    novapp::Range const rng({0, 0, 0}, {10, 11, 12}, 1);
    novapp::KDV_double_3d array("array", 12, 11, 12);
    Kokkos::deep_copy(array.d_view, 0.);
    array.modify_device();
    novapp::broadcast(rng.no_ghosts(), 5.5, array.d_view);
    array.modify_device();
    array.sync_host();
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 11; ++j) {
            for (int k = 0; k < 12; ++k) {
                if ((i == 0 || i == 11)) {
                    EXPECT_DOUBLE_EQ(array.h_view(i, j, k), 0.);
                } else {
                    EXPECT_DOUBLE_EQ(array.h_view(i, j, k), 5.5);
                }
            }
        }
    }
}

TEST(BroadcastScalar, Case2d)
{
    if (novapp::ndim != 2) {
        GTEST_SKIP();
    }
    novapp::Range const rng({0, 0, 0}, {10, 11, 12}, 1);
    novapp::KDV_double_3d array("array", 12, 13, 12);
    Kokkos::deep_copy(array.d_view, 0.);
    array.modify_device();
    novapp::broadcast(rng.no_ghosts(), 5.5, array.d_view);
    array.modify_device();
    array.sync_host();
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 13; ++j) {
            for (int k = 0; k < 12; ++k) {
                if ((i == 0 || i == 11) || (j == 0 || j == 12)) {
                    EXPECT_DOUBLE_EQ(array.h_view(i, j, k), 0.);
                } else {
                    EXPECT_DOUBLE_EQ(array.h_view(i, j, k), 5.5);
                }
            }
        }
    }
}

TEST(BroadcastScalar, Case3d)
{
    if (novapp::ndim != 3) {
        GTEST_SKIP();
    }
    novapp::Range const rng({0, 0, 0}, {10, 11, 12}, 1);
    novapp::KDV_double_3d array("array", 12, 13, 14);
    Kokkos::deep_copy(array.d_view, 0.);
    array.modify_device();
    novapp::broadcast(rng.no_ghosts(), 5.5, array.d_view);
    array.modify_device();
    array.sync_host();
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 13; ++j) {
            for (int k = 0; k < 14; ++k) {
                if ((i == 0 || i == 11) || (j == 0 || j == 12) || (k == 0 || k == 13)) {
                    EXPECT_DOUBLE_EQ(array.h_view(i, j, k), 0.);
                } else {
                    EXPECT_DOUBLE_EQ(array.h_view(i, j, k), 5.5);
                }
            }
        }
    }
}

TEST(BroadcastArray, Case1d)
{
    if (novapp::ndim != 1) {
        GTEST_SKIP();
    }
    novapp::Range const rng({0, 0, 0}, {10, 11, 12}, 1);
    novapp::KDV_double_3d array("array", 12, 11, 12);
    Kokkos::deep_copy(array.d_view, 0.);
    array.modify_device();
    novapp::KV_double_1d const arr("arr", 10);
    Kokkos::deep_copy(arr, 5.5);
    novapp::broadcast(rng.no_ghosts(), arr, array.d_view);
    array.modify_device();
    array.sync_host();
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 11; ++j) {
            for (int k = 0; k < 12; ++k) {
                if ((i == 0 || i == 11)) {
                    EXPECT_DOUBLE_EQ(array.h_view(i, j, k), 0.);
                } else {
                    EXPECT_DOUBLE_EQ(array.h_view(i, j, k), 5.5);
                }
            }
        }
    }
}

TEST(BroadcastArray, Case2d)
{
    if (novapp::ndim != 2) {
        GTEST_SKIP();
    }
    novapp::Range const rng({0, 0, 0}, {10, 11, 12}, 1);
    novapp::KDV_double_3d array("array", 12, 13, 12);
    Kokkos::deep_copy(array.d_view, 0.);
    array.modify_device();
    novapp::KV_double_1d const arr("arr", 10);
    Kokkos::deep_copy(arr, 5.5);
    novapp::broadcast(rng.no_ghosts(), arr, array.d_view);
    array.modify_device();
    array.sync_host();
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 13; ++j) {
            for (int k = 0; k < 12; ++k) {
                if ((i == 0 || i == 11) || (j == 0 || j == 12)) {
                    EXPECT_DOUBLE_EQ(array.h_view(i, j, k), 0.);
                } else {
                    EXPECT_DOUBLE_EQ(array.h_view(i, j, k), 5.5);
                }
            }
        }
    }
}

TEST(BroadcastArray, Case3d)
{
    if (novapp::ndim != 3) {
        GTEST_SKIP();
    }
    novapp::Range const rng({0, 0, 0}, {10, 11, 12}, 1);
    novapp::KDV_double_3d array("array", 12, 13, 14);
    Kokkos::deep_copy(array.d_view, 0.);
    array.modify_device();
    novapp::KV_double_1d const arr("arr", 10);
    Kokkos::deep_copy(arr, 5.5);
    novapp::broadcast(rng.no_ghosts(), arr, array.d_view);
    array.modify_device();
    array.sync_host();
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 13; ++j) {
            for (int k = 0; k < 14; ++k) {
                if ((i == 0 || i == 11) || (j == 0 || j == 12) || (k == 0 || k == 13)) {
                    EXPECT_DOUBLE_EQ(array.h_view(i, j, k), 0.);
                } else {
                    EXPECT_DOUBLE_EQ(array.h_view(i, j, k), 5.5);
                }
            }
        }
    }
}
