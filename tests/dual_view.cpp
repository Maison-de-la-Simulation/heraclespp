// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <dual_view.hpp>

TEST(DualView, Constructor)
{
    hclpp::DualView<int**> const dv("label", 10, 13);
    EXPECT_EQ(dv.rank, 2);
    EXPECT_EQ(dv.rank_dynamic, 2);
    EXPECT_EQ(dv.label(), "label");
    EXPECT_EQ(dv.extent(0), 10);
    EXPECT_EQ(dv.extent(1), 13);
    EXPECT_EQ(dv.stride(0), 13);
    EXPECT_EQ(dv.stride(1), 1);
    EXPECT_EQ(dv.span(), 130);
    EXPECT_TRUE(dv.span_is_contiguous());
}

TEST(DualView, Access)
{
    {
        hclpp::DualView<int**> const dv("label", 10, 10);
        [[maybe_unused]] auto a = dv(hclpp::device_read_write);
        EXPECT_FALSE(dv.need_synchronization(hclpp::device));
        EXPECT_TRUE(dv.need_synchronization(hclpp::host));
        dv.synchronize(hclpp::host);
        EXPECT_FALSE(dv.need_synchronization(hclpp::device));
        EXPECT_FALSE(dv.need_synchronization(hclpp::host));
        [[maybe_unused]] auto b = dv(hclpp::device_read_only);
        EXPECT_FALSE(dv.need_synchronization(hclpp::device));
        EXPECT_FALSE(dv.need_synchronization(hclpp::host));
        [[maybe_unused]] auto c = dv(hclpp::device_read_write);
        EXPECT_FALSE(dv.need_synchronization(hclpp::device));
        EXPECT_TRUE(dv.need_synchronization(hclpp::host));
    }
    {
        hclpp::DualView<int**> const dv("label", 10, 10);
        hclpp::DualView<int const**> const cdv(dv);
        [[maybe_unused]] auto b = cdv(hclpp::device_read_only);
    }
}

TEST(DualViewBasic, HostWriteThenDeviceRead)
{
    int const ext0 = 16;

    hclpp::DualView<int*> const dv("dv_hw_dr", ext0);
    EXPECT_TRUE(dv.is_allocated());

    {
        auto const host_wa = dv(hclpp::host_discard_write);
        for (int i = 0; i < ext0; ++i) {
            host_wa(i) = i;
        }
    }

    EXPECT_TRUE(dv.need_synchronization(hclpp::device));
    dv.synchronize(hclpp::device);
    EXPECT_FALSE(dv.need_synchronization(hclpp::device));

    {
        hclpp::DualView<int*>::view_host_type const mirror("mirror", ext0);
        Kokkos::deep_copy(mirror, dv(hclpp::device_read_only));
        for (int i = 0; i < ext0; ++i) {
            EXPECT_EQ(mirror(i), i);
        }
    }
}

TEST(DualViewBasic, DeviceWriteThenHostReadAndClear)
{
    int const ext0 = 16;

    hclpp::DualView<int*> const dv("dv_dw_hr", ext0);
    EXPECT_TRUE(dv.is_allocated());

    {
        // modify device
        hclpp::DualView<int*>::view_host_type const host_init("host_init", ext0);
        for (int i = 0; i < ext0; ++i) {
            host_init(i) = i;
        }
        Kokkos::deep_copy(dv(hclpp::device_discard_write), host_init);
    }

    EXPECT_TRUE(dv.need_synchronization(hclpp::host));

    {
        auto const host_ro = dv(hclpp::host_read_only);
        for (int i = 0; i < ext0; ++i) {
            EXPECT_EQ(host_ro(i), i);
        }
    }

    EXPECT_FALSE(dv.need_synchronization(hclpp::host));

    // clear sync state
    dv.clear_synchronization_state();
    EXPECT_FALSE(dv.need_synchronization(hclpp::host));
    EXPECT_FALSE(dv.need_synchronization(hclpp::device));
}
