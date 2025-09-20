# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

add_library(mesh)
target_link_libraries(mesh PUBLIC Kokkos::kokkos MPI::MPI_CXX utils)
target_sources(
    mesh
    PRIVATE
        geometry.cpp
        geometry_factory.cpp
        grid.cpp
        grid_factory.cpp
        grid_type.cpp
        range.cpp
    INTERFACE
        FILE_SET HEADERS
            FILES
                geometry_factory.hpp
                geometry.hpp
                grid_factory.hpp
                grid_type.hpp
                grid.hpp
                range.hpp
)
