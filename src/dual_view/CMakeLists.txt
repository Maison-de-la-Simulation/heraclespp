# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

add_library(dual_view INTERFACE)
target_compile_features(dual_view INTERFACE cxx_std_17)
target_link_libraries(dual_view INTERFACE Kokkos::kokkos)
target_sources(dual_view INTERFACE FILE_SET HEADERS FILES dual_view.hpp)
