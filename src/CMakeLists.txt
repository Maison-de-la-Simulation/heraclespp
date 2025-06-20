# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

# Workaround a defect from FindHDF5.cmake
if(
    ("${Novapp_inih_DEPENDENCY_POLICY}" STREQUAL "INSTALLED")
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

add_library(
    libnova++
    boundary.cpp
    boundary_distribute.cpp
    broadcast.cpp
    initialization_interface.cpp
    mpi_scope_guard.cpp
    shift_criterion_interface.cpp
)
target_compile_features(libnova++ PUBLIC cxx_std_17)
target_include_directories(libnova++ PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    libnova++
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

add_executable(nova++ main.cpp)
target_compile_definitions(
    nova++
    PUBLIC MY_SETUP="${Novapp_SETUP}" NOVAPP_GRAVITY_${Novapp_GRAVITY}
)
target_compile_features(nova++ PUBLIC cxx_std_20)
target_include_directories(nova++ PUBLIC setups/${Novapp_SETUP}/)
target_link_libraries(
    nova++
    PUBLIC inih io libnova++ MPI::MPI_CXX paraconf::paraconf PDI::PDI_C
)

include(GNUInstallDirs)

install(TARGETS nova++ RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
