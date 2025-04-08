# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

add_library(analysis integration.cpp internal_energy.cpp)
target_include_directories(analysis PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    analysis
    PRIVATE MPI::MPI_CXX
    PUBLIC hydro Kokkos::kokkos mesh utils
)
