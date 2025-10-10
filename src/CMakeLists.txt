# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

find_package(MPI REQUIRED COMPONENTS CXX)

add_subdirectory(analysis)
add_subdirectory(gravity)
add_subdirectory(hydro)
add_subdirectory(mesh)
add_subdirectory(thermodynamics)
add_subdirectory(units)
add_subdirectory(user_step)
add_subdirectory(utils)
add_subdirectory(moving_grid)

add_library(libnova++)
target_compile_features(libnova++ PUBLIC cxx_std_17)
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
target_sources(
    libnova++
    PRIVATE
        boundary.cpp
        boundary_distribute.cpp
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

add_executable(nova++ main.cpp)
target_compile_definitions(
    nova++
    PUBLIC MY_SETUP="${Novapp_SETUP}" NOVAPP_GRAVITY_${Novapp_GRAVITY}
)
target_compile_features(nova++ PUBLIC cxx_std_20)
target_include_directories(nova++ PUBLIC setups/${Novapp_SETUP}/)
target_link_libraries(
    nova++
    PUBLIC inih libnova++ MPI::MPI_CXX
)

include(GNUInstallDirs)

install(TARGETS nova++ RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
