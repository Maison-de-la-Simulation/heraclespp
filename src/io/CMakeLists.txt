# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

add_library(io)
target_link_libraries(
    io
    PRIVATE mesh MPI::MPI_CXX PDI::PDI_C
    PUBLIC hdf5::hdf5 utils
)
target_sources(
    io
    PRIVATE io.cpp config.yaml.cpp
    INTERFACE FILE_SET HEADERS FILES config.yaml.hpp io_hdf5.hpp io.hpp
)
