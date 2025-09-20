# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

add_library(thermodynamics)
target_compile_features(thermodynamics PUBLIC cxx_std_17)
target_link_libraries(thermodynamics PUBLIC Kokkos::kokkos units)
target_sources(
    thermodynamics
    PRIVATE PerfectGas.cpp RadGas.cpp
    INTERFACE FILE_SET HEADERS FILES PerfectGas.hpp RadGas.hpp
)
