// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see
// COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <int_cast.hpp>
#include <limits>
#include <stdexcept>

TEST(IntCast, UnsignedToInt) {
  ASSERT_EQ(novapp::int_cast<int>(0U), 0);
  ASSERT_EQ(novapp::int_cast<int>(1U), 1);
  ASSERT_THROW(novapp::int_cast<int>(std::numeric_limits<unsigned>::max()),
               std::runtime_error);
}

TEST(IntCast, IntToUnsigned) {
  ASSERT_EQ(novapp::int_cast<unsigned>(0), 0U);
  ASSERT_EQ(novapp::int_cast<unsigned>(1), 1U);
  ASSERT_THROW(novapp::int_cast<unsigned>(-1), std::runtime_error);
}
