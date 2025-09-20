# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

add_library(units INTERFACE)
target_link_libraries(units INTERFACE Kokkos::kokkos)
target_sources(units INTERFACE FILE_SET HEADERS FILES units.hpp)
