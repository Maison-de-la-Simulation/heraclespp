# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

add_library(units)
target_link_libraries(units PUBLIC Kokkos::kokkos)
target_sources(
    units
    PRIVATE units.cpp
    INTERFACE FILE_SET HEADERS FILES units.hpp
)
