# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

add_library(thermodynamics PerfectGas.cpp RadGas.cpp)
target_compile_features(thermodynamics PUBLIC cxx_std_17)
target_include_directories(thermodynamics INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(thermodynamics PUBLIC Kokkos::kokkos units)
