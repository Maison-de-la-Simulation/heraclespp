include(GoogleTest)

add_executable(unit_tests main.cpp conversions.cpp euler_equations.cpp grid_type.cpp kronecker.cpp partitioner.cpp range.cpp riemann_solvers.cpp slopes.cpp thermodynamics.cpp)
target_compile_features(unit_tests PUBLIC cxx_std_17)
target_link_libraries(unit_tests PUBLIC libnova++ GTest::gtest)

gtest_discover_tests(unit_tests)
