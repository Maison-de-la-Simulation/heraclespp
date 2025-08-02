# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

add_library(gravity gravity.cpp)
target_compile_features(gravity PUBLIC cxx_std_20)
target_include_directories(gravity INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    gravity
    PUBLIC Kokkos::kokkos utils
    PRIVATE mesh MPI::MPI_CXX units
)
