# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

add_library(gravity)
target_compile_features(gravity PUBLIC cxx_std_20)
target_link_libraries(
    gravity
    PUBLIC Kokkos::kokkos utils
    PRIVATE dual_view mesh MPI::MPI_CXX units
)
target_sources(
    gravity
    PRIVATE gravity.cpp
    INTERFACE FILE_SET HEADERS FILES gravity.hpp
)
