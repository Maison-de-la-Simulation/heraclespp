# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

add_library(io io.cpp config.yaml.cpp)
target_include_directories(io INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    io
    PRIVATE mesh MPI::MPI_CXX PDI::PDI_C
    PUBLIC hdf5::hdf5 utils
)
