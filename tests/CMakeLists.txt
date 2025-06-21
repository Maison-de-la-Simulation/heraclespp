# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

include(GoogleTest)

add_executable(
    unit_tests
    main.cpp
    broadcast.cpp
    conversions.cpp
    kokkos_shortcut.cpp
    euler_equations.cpp
    gravity.cpp
    grid_type.cpp
    kronecker.cpp
    partitioner.cpp
    print_info.cpp
    range.cpp
    riemann_solvers.cpp
    slopes.cpp
    thermodynamics.cpp
    int_cast.cpp
)
target_compile_features(unit_tests PUBLIC cxx_std_17)
target_link_libraries(unit_tests PUBLIC libnova++ GTest::gtest)

gtest_discover_tests(unit_tests DISCOVERY_MODE PRE_TEST)
