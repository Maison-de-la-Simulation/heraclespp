# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

add_library(hydro face_reconstruction.cpp)
target_compile_features(hydro PUBLIC cxx_std_20)
target_include_directories(hydro INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    hydro
    PUBLIC Kokkos::kokkos mesh utils
    INTERFACE MPI::MPI_CXX
)
