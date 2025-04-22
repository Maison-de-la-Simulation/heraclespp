# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

add_library(gravity INTERFACE)
target_include_directories(gravity INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gravity INTERFACE Kokkos::kokkos mesh units utils)
