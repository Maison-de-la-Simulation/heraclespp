// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <sstream>

#include <gtest/gtest.h>

#include <print_info.hpp>

namespace novapp {

TEST(PrintInfo, Format)
{
    std::stringstream ss;
    novapp::print_info(ss, "Parameter", std::string_view("Value"));
    EXPECT_EQ(ss.str(), "Parameter..................................................................Value\n");
}

} // namespace novapp
