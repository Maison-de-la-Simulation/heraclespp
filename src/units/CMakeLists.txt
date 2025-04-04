# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

add_library(units INTERFACE)
target_include_directories(units INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(units INTERFACE Kokkos::kokkos)
