// SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <sstream>
#include <string_view>

#include <gtest/gtest.h>

#include <print_info.hpp>

namespace hclpp {

TEST(PrintInfo, Format)
{
    std::stringstream ss;
    hclpp::print_info(ss, "Parameter", std::string_view("Value"));
    EXPECT_EQ(ss.str(), "Parameter..................................................................Value\n");
}

} // namespace hclpp
