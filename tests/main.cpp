#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    Kokkos::ScopeGuard const kokkos_scope(argc, argv);

    return RUN_ALL_TESTS();
}
