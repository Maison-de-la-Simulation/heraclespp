# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

add_library(analysis)
target_compile_features(analysis INTERFACE cxx_std_20)
target_link_libraries(
    analysis
    PRIVATE MPI::MPI_CXX
    PUBLIC hydro Kokkos::kokkos mesh utils
)
target_sources(
    analysis
    PRIVATE integration.cpp internal_energy.cpp
    PUBLIC
        FILE_SET HEADERS
            FILES integration.hpp internal_energy.hpp temperature.hpp
)
