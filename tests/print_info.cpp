#include <sstream>

#include <gtest/gtest.h>

#include <print_info.hpp>

namespace novapp {

TEST(PrintInfo, Format)
{
    std::stringstream ss;
    novapp::print_info(ss, "Parameter", "Value");
    EXPECT_EQ(ss.str(), "Parameter..................................................................Value\n");
}

} // namespace novapp