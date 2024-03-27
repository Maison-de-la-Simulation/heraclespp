#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

namespace testing::internal {

// accessing gtest internals is not very clean, but gtest provides no public access...
extern bool g_help_flag;

} // namespace testing::internal

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    if (::testing::GTEST_FLAG(list_tests) || ::testing::internal::g_help_flag) {
        // do not initialize Kokkos just to list tests or show help so as to be able to run
        // a Cuda/Hip/etc. enabled with no device available
        return RUN_ALL_TESTS();
    }

    Kokkos::ScopeGuard const kokkos_scope(argc, argv);

    return RUN_ALL_TESTS();
}
