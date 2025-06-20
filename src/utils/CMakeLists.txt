# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

find_package(Git QUIET)
if("${GIT_FOUND}")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" describe --tags --always --dirty
        OUTPUT_VARIABLE GIT_BUILD_STRING
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        OUTPUT_VARIABLE GIT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

if("${GIT_BUILD_STRING}" STREQUAL "")
    set(GIT_BUILD_STRING "N/A")
endif()
if("${GIT_BRANCH}" STREQUAL "")
    set(GIT_BRANCH "N/A")
endif()

execute_process(
    COMMAND date "+%d/%m/%y"
    OUTPUT_VARIABLE DATE_STRING
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND date "+%H:%M:%S"
    OUTPUT_VARIABLE TIME_STRING
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

add_library(utils nova_params.cpp ${CMAKE_CURRENT_BINARY_DIR}/git_version.cpp)
target_include_directories(
    utils
    INTERFACE ${CMAKE_CURRENT_BINARY_DIR}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_features(utils PUBLIC cxx_std_20)
target_link_libraries(utils INTERFACE Kokkos::kokkos thermodynamics PUBLIC inih)

configure_file(eos.hpp.in eos.hpp NO_SOURCE_PERMISSIONS @ONLY)
configure_file(geom.hpp.in geom.hpp NO_SOURCE_PERMISSIONS @ONLY)
configure_file(ndim.hpp.in ndim.hpp NO_SOURCE_PERMISSIONS @ONLY)
configure_file(git_version.cpp.in git_version.cpp NO_SOURCE_PERMISSIONS @ONLY)
