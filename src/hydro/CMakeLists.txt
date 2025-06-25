# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

add_library(hydro INTERFACE)
target_compile_features(hydro INTERFACE cxx_std_20)
target_include_directories(hydro INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(hydro INTERFACE Kokkos::kokkos mesh MPI::MPI_CXX utils)
