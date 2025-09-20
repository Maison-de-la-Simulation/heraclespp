# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

add_library(user_step)
target_link_libraries(user_step PRIVATE Kokkos::kokkos mesh units PUBLIC utils)
target_sources(
    user_step
    PRIVATE user_step.cpp user_step_factory.cpp
    INTERFACE FILE_SET HEADERS FILES user_step_factory.hpp user_step.hpp
)
