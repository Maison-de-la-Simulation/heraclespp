# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

add_library(moving_grid)
target_compile_features(moving_grid PUBLIC cxx_std_17)
target_link_libraries(moving_grid PRIVATE Kokkos::kokkos mesh PUBLIC utils)
target_sources(
    moving_grid
    PRIVATE moving_grid.cpp
    INTERFACE FILE_SET HEADERS FILES moving_grid.hpp
)
