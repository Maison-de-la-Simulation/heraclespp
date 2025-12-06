# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

# Workaround a defect from FindHDF5.cmake
if(
    ("${Heraclespp_inih_DEPENDENCY_POLICY}" STREQUAL "INSTALLED")
    AND (CMAKE_VERSION VERSION_LESS "3.26.1")
)
    enable_language(C)
endif()
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(paraconf REQUIRED COMPONENTS C)
find_package(PDI REQUIRED COMPONENTS C)

add_subdirectory(analysis)
add_subdirectory(gravity)
add_subdirectory(hydro)
add_subdirectory(io)
add_subdirectory(mesh)
add_subdirectory(thermodynamics)
add_subdirectory(units)
add_subdirectory(user_step)
add_subdirectory(utils)
add_subdirectory(moving_grid)

add_library(libheracles++)
target_compile_features(libheracles++ PUBLIC cxx_std_17)
target_link_libraries(
    libheracles++
    PRIVATE MPI::MPI_CXX
    PUBLIC
        analysis
        gravity
        hydro
        Kokkos::kokkos
        mesh
        moving_grid
        thermodynamics
        units
        user_step
        utils
)
target_sources(
    libheracles++
    PRIVATE
        boundary.cpp
        boundary_distribute.cpp
        boundary_factory.cpp
        broadcast.cpp
        initialization_interface.cpp
        mpi_scope_guard.cpp
        shift_criterion_interface.cpp
    PUBLIC
        FILE_SET HEADERS
            FILES
                boundary_distribute.hpp
                boundary_factory.hpp
                boundary.hpp
                broadcast.hpp
                initialization_interface.hpp
                mpi_scope_guard.hpp
                shift_criterion_interface.hpp
)

add_executable(heracles++ main.cpp setups/${Heraclespp_SETUP}/setup.cpp)
target_compile_definitions(
    heracles++
    PUBLIC
        MY_SETUP="${Heraclespp_SETUP}"
        HERACLESPP_GRAVITY_${Heraclespp_GRAVITY}
)
target_compile_features(heracles++ PUBLIC cxx_std_20)
target_include_directories(heracles++ PUBLIC setups/${Heraclespp_SETUP}/)
target_link_libraries(
    heracles++
    PUBLIC inih io libheracles++ MPI::MPI_CXX paraconf::paraconf PDI::PDI_C
)

include(GNUInstallDirs)

install(TARGETS heracles++ RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
