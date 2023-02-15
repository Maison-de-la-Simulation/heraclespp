include(GoogleTest)

add_executable(unit_tests main.cpp conversions.cpp euler_equations.cpp perfect_gas.cpp riemann_solvers.cpp slopes.cpp)
target_compile_features(unit_tests PUBLIC cxx_std_17)
target_link_libraries(unit_tests PUBLIC libnova++ GTest::gtest)

gtest_discover_tests(unit_tests)
