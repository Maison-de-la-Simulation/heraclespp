# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

add_library(thermodynamics)
target_compile_features(thermodynamics PUBLIC cxx_std_17)
target_link_libraries(thermodynamics PUBLIC Kokkos::kokkos units)
target_sources(
    thermodynamics
    PRIVATE perfect_gas.cpp rad_gas.cpp
    INTERFACE FILE_SET HEADERS FILES perfect_gas.hpp rad_gas.hpp
)
