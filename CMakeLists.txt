# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)

project(HERACLES++ LANGUAGES CXX)

set_property(GLOBAL PROPERTY GLOBAL_DEPENDS_NO_CYCLES ON)

option(Heraclespp_BUILD_BENCHMARKING "Build the benchmarking tree." OFF)

set(Heraclespp_EOS
    ""
    CACHE STRING
    "Choose the equation of state: PerfectGas(Perfect Gas) ; Gas+Radiation(Gas+Radiation)"
)
set_property(CACHE Heraclespp_EOS PROPERTY STRINGS "PerfectGas;Gas+Radiation")
if("${Heraclespp_EOS}" STREQUAL "")
    message(FATAL_ERROR "Heraclespp_EOS is not defined")
endif()

set(Heraclespp_GEOM
    ""
    CACHE STRING
    "Choose the geometric system: Cartesian(Cartesian system) ; Spherical(Spherical system)"
)
set_property(CACHE Heraclespp_GEOM PROPERTY STRINGS "Cartesian;Spherical")
if("${Heraclespp_GEOM}" STREQUAL "")
    message(FATAL_ERROR "Heraclespp_GEOM is not defined")
endif()

set(Heraclespp_GRAVITY
    ""
    CACHE STRING
    "Choose the equation of state: Uniform(Uniform gravity) ; Point_mass(Point mass gravity); Internal_Mass(Internal mass gravity)"
)
set_property(
    CACHE Heraclespp_GRAVITY
    PROPERTY STRINGS "Uniform;Point_mass;Internal_mass"
)
if("${Heraclespp_GRAVITY}" STREQUAL "")
    message(FATAL_ERROR "Heraclespp_GRAVITY is not defined")
endif()

set(DIMS "1;2;3")
set(Heraclespp_NDIM "" CACHE STRING "Choose the dimension, options are ${DIMS}")
set_property(CACHE Heraclespp_NDIM PROPERTY STRINGS "${DIMS}")
if(NOT ("${Heraclespp_NDIM}" IN_LIST DIMS))
    message(
        FATAL_ERROR
        "Heraclespp_NDIM is not defined, available dimensions are ${DIMS}"
    )
endif()

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(
        FATAL_ERROR
        "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. "
    )
endif()

set(Heraclespp_SETUP "" CACHE STRING "Choose the setup")
file(
    GLOB sub-dirs-name
    RELATIVE ${PROJECT_SOURCE_DIR}/src/setups
    ${PROJECT_SOURCE_DIR}/src/setups/*
)
set_property(CACHE Heraclespp_SETUP PROPERTY STRINGS "${sub-dirs-name}")

if(NOT ("${Heraclespp_SETUP}" IN_LIST sub-dirs-name))
    message(
        FATAL_ERROR
        "Heraclespp_SETUP is not defined, available setups are: ${sub-dirs-name}"
    )
endif()

include(CTest)

set(Heraclespp_DEPENDENCY_POLICIES "EMBEDDED" "INSTALLED")
set(Heraclespp_DEFAULT_DEPENDENCY_POLICY "EMBEDDED")
if(DEFINED ENV{HERACLESPP_DEFAULT_DEPENDENCY_POLICY})
    set(Heraclespp_DEFAULT_DEPENDENCY_POLICY
        "$ENV{HERACLESPP_DEFAULT_DEPENDENCY_POLICY}"
    )
endif()
if(DEFINED CACHE{Heraclespp_DEFAULT_DEPENDENCY_POLICY})
    set(Heraclespp_DEFAULT_DEPENDENCY_POLICY
        "$CACHE{Heraclespp_DEFAULT_DEPENDENCY_POLICY}"
    )
endif()

# benchmark dependency
set(Heraclespp_benchmark_DEPENDENCY_POLICY
    "${Heraclespp_DEFAULT_DEPENDENCY_POLICY}"
    CACHE STRING
    "Policy to find the `benchmark` package. Options: ${Heraclespp_DEPENDENCY_POLICIES}"
)
set_property(
    CACHE Heraclespp_benchmark_DEPENDENCY_POLICY
    PROPERTY STRINGS "${Heraclespp_DEPENDENCY_POLICIES}"
)
if("${Heraclespp_benchmark_DEPENDENCY_POLICY}" STREQUAL "EMBEDDED")
    add_subdirectory(vendor/benchmark SYSTEM)
elseif("${Heraclespp_benchmark_DEPENDENCY_POLICY}" STREQUAL "INSTALLED")
    find_package(benchmark REQUIRED)
endif()

# Doxygen dependency
find_package(Doxygen)

# GTest dependency
set(Heraclespp_GTest_DEPENDENCY_POLICY
    "${Heraclespp_DEFAULT_DEPENDENCY_POLICY}"
    CACHE STRING
    "Policy to find the `GTest` package. Options: ${Heraclespp_DEPENDENCY_POLICIES}"
)
set_property(
    CACHE Heraclespp_GTest_DEPENDENCY_POLICY
    PROPERTY STRINGS "${Heraclespp_DEPENDENCY_POLICIES}"
)
if("${Heraclespp_GTest_DEPENDENCY_POLICY}" STREQUAL "EMBEDDED")
    add_subdirectory(vendor/googletest SYSTEM)
elseif("${Heraclespp_GTest_DEPENDENCY_POLICY}" STREQUAL "INSTALLED")
    find_package(GTest REQUIRED CONFIG)
endif()

# inih dependency
set(Heraclespp_inih_DEPENDENCY_POLICY
    "${Heraclespp_DEFAULT_DEPENDENCY_POLICY}"
    CACHE STRING
    "Policy to find the `inih` package. Options: ${Heraclespp_DEPENDENCY_POLICIES}"
)
set_property(
    CACHE Heraclespp_inih_DEPENDENCY_POLICY
    PROPERTY STRINGS "${Heraclespp_DEPENDENCY_POLICIES}"
)
if("${Heraclespp_inih_DEPENDENCY_POLICY}" STREQUAL "EMBEDDED")
    add_subdirectory(vendor/inih SYSTEM)
elseif("${Heraclespp_inih_DEPENDENCY_POLICY}" STREQUAL "INSTALLED")
    find_package(inih REQUIRED CONFIG)
endif()

# Kokkos dependency
set(Heraclespp_Kokkos_DEPENDENCY_POLICY
    "${Heraclespp_DEFAULT_DEPENDENCY_POLICY}"
    CACHE STRING
    "Policy to find the `Kokkos` package. Options: ${Heraclespp_DEPENDENCY_POLICIES}"
)
set_property(
    CACHE Heraclespp_Kokkos_DEPENDENCY_POLICY
    PROPERTY STRINGS "${Heraclespp_DEPENDENCY_POLICIES}"
)
if("${Heraclespp_Kokkos_DEPENDENCY_POLICY}" STREQUAL "EMBEDDED")
    add_subdirectory(vendor/kokkos SYSTEM)
elseif("${Heraclespp_Kokkos_DEPENDENCY_POLICY}" STREQUAL "INSTALLED")
    find_package(Kokkos 4...<6 REQUIRED CONFIG)
endif()

add_subdirectory(src)

if("${Doxygen_FOUND}")
    set(DOXYGEN_EXCLUDE_PATTERNS "*/deprecated/*")
    set(DOXYGEN_EXTRACT_ALL YES)
    set(DOXYGEN_WARN_NO_PARAMDOC YES)
    set(DOXYGEN_WARN_IF_UNDOCUMENTED NO)
    doxygen_add_docs(doxygen ${PROJECT_SOURCE_DIR}/src)
endif()

if("${BUILD_TESTING}")
    add_subdirectory(tests)
endif()

if("${Heraclespp_BUILD_BENCHMARKING}")
    add_subdirectory(benchmarks)
endif()
