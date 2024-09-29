add_library(thermodynamics PerfectGas.cpp RadGas.cpp)

target_link_libraries(thermodynamics PUBLIC Kokkos::kokkos units)
target_compile_features(thermodynamics PUBLIC cxx_std_17)
target_include_directories(thermodynamics INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
