// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Kokkos_Array.hpp>
#include <kronecker.hpp>

TEST(Kronecker, Kronecker)
{
    ASSERT_EQ(novapp::kron(1, 1), 1);
    ASSERT_EQ(novapp::kron(1, 2), 0);
    ASSERT_EQ(novapp::kron(2, 1), 0);
    ASSERT_EQ(novapp::kron(2, 2), 1);
}

TEST(Kronecker, LIndex)
{
    {
        auto const [i, j, k] = novapp::lindex(0, 2, 1, 1);
        ASSERT_EQ(i, 1);
        ASSERT_EQ(j, 1);
        ASSERT_EQ(k, 1);
    }

    {
        auto const [i, j, k] = novapp::lindex(1, 1, 2, 1);
        ASSERT_EQ(i, 1);
        ASSERT_EQ(j, 1);
        ASSERT_EQ(k, 1);
    }

    {
        auto const [i, j, k] = novapp::lindex(2, 1, 1, 2);
        ASSERT_EQ(i, 1);
        ASSERT_EQ(j, 1);
        ASSERT_EQ(k, 1);
    }
}

TEST(Kronecker, RIndex)
{
    {
        auto const [i, j, k] = novapp::rindex(0, 0, 1, 1);
        ASSERT_EQ(i, 1);
        ASSERT_EQ(j, 1);
        ASSERT_EQ(k, 1);
    }

    {
        auto const [i, j, k] = novapp::rindex(1, 1, 0, 1);
        ASSERT_EQ(i, 1);
        ASSERT_EQ(j, 1);
        ASSERT_EQ(k, 1);
    }

    {
        auto const [i, j, k] = novapp::rindex(2, 1, 1, 0);
        ASSERT_EQ(i, 1);
        ASSERT_EQ(j, 1);
        ASSERT_EQ(k, 1);
    }
}
