# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

add_library(moving_grid moving_grid.cpp)
target_compile_features(moving_grid PUBLIC cxx_std_17)
target_include_directories(moving_grid INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(moving_grid PRIVATE Kokkos::kokkos mesh PUBLIC utils)
