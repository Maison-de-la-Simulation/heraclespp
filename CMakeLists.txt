cmake_minimum_required(VERSION 3.20)

project(Nova++ LANGUAGES CXX)

set_property(GLOBAL PROPERTY GLOBAL_DEPENDS_NO_CYCLES ON)

set(Novapp_EOS "" CACHE STRING "Choose the equation of state: PerfectGas(Perfect Gas) ; Gas+Radiation(Gas+Radiation)")
set_property(CACHE Novapp_EOS PROPERTY STRINGS "PerfectGas;Gas+Radiation")
if("${Novapp_EOS}" STREQUAL "")
  message(FATAL_ERROR "Novapp_EOS is not defined")
endif()

set(Novapp_GEOM "" CACHE STRING "Choose the geometric system: Cartesian(Cartesian system) ; Spherical(Spherical system)")
set_property(CACHE Novapp_GEOM PROPERTY STRINGS "Cartesian;Spherical")
if("${Novapp_GEOM}" STREQUAL "")
  message(FATAL_ERROR "Novapp_GEOM is not defined")
endif()

set(Novapp_GRAVITY "" CACHE STRING "Choose the equation of state: Uniform(Uniform gravity) ; Point_mass(Point mass gravity)")
set_property(CACHE Novapp_GRAVITY PROPERTY STRINGS "Uniform;Point_mass")
if("${Novapp_GRAVITY}" STREQUAL "")
  message(FATAL_ERROR "Novapp_GRAVITY is not defined")
endif()

set(DIMS "1;2;3")
set(Novapp_NDIM "" CACHE STRING "Choose the dimension, options are ${DIMS}")
set_property(CACHE Novapp_NDIM PROPERTY STRINGS "${DIMS}")
if(NOT ("${Novapp_NDIM}"  IN_LIST DIMS))
  message(FATAL_ERROR "Novapp_NDIM is not defined, available dimensions are ${DIMS}")
endif()

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. ")
endif()

set(Novapp_SETUP "" CACHE STRING "Choose the setup")
file(GLOB sub-dirs-name  RELATIVE ${PROJECT_SOURCE_DIR}/src/setups ${PROJECT_SOURCE_DIR}/src/setups/*)
set_property(CACHE Novapp_SETUP PROPERTY STRINGS "${sub-dirs-name}")

if (NOT("${Novapp_SETUP}" IN_LIST sub-dirs-name))
  message(FATAL_ERROR "Novapp_SETUP is not defined, available setups are: ${sub-dirs-name}")
endif()

include(CTest)

set(Novapp_DEPENDENCY_POLICIES "EMBEDDED" "INSTALLED")

# Doxygen dependency
find_package(Doxygen)

# GTest dependency
set(Novapp_GTest_DEPENDENCY_POLICY "EMBEDDED" CACHE STRING "Policy to find the `GTest` package. Options: ${Novapp_DEPENDENCY_POLICIES}")
set_property(CACHE Novapp_GTest_DEPENDENCY_POLICY PROPERTY STRINGS "${Novapp_DEPENDENCY_POLICIES}")
if("${Novapp_GTest_DEPENDENCY_POLICY}" STREQUAL "EMBEDDED")
  add_subdirectory(vendor/googletest)
elseif("${Novapp_GTest_DEPENDENCY_POLICY}" STREQUAL "INSTALLED")
  find_package(GTest REQUIRED CONFIG)
endif()

# inih dependency
set(Novapp_inih_DEPENDENCY_POLICY "EMBEDDED" CACHE STRING "Policy to find the `inih` package. Options: EMBEDDED")
set_property(CACHE Novapp_inih_DEPENDENCY_POLICY PROPERTY STRINGS "EMBEDDED")
if("${Novapp_inih_DEPENDENCY_POLICY}" STREQUAL "EMBEDDED")
  add_subdirectory(vendor/inih)
endif()

# Kokkos dependency
set(Novapp_Kokkos_DEPENDENCY_POLICY "EMBEDDED" CACHE STRING "Policy to find the `Kokkos` package. Options: ${Novapp_DEPENDENCY_POLICIES}")
set_property(CACHE Novapp_Kokkos_DEPENDENCY_POLICY PROPERTY STRINGS "${Novapp_DEPENDENCY_POLICIES}")
if("${Novapp_Kokkos_DEPENDENCY_POLICY}" STREQUAL "EMBEDDED")
  add_subdirectory(vendor/kokkos)
elseif("${Novapp_Kokkos_DEPENDENCY_POLICY}" STREQUAL "INSTALLED")
  find_package(Kokkos 4...<5 REQUIRED CONFIG)
endif()

add_subdirectory(src)

if ("${Doxygen_FOUND}")
  set(DOXYGEN_EXCLUDE_PATTERNS "*/deprecated/*")
  set(DOXYGEN_EXTRACT_ALL YES)
  set(DOXYGEN_WARN_NO_PARAMDOC YES)
  set(DOXYGEN_WARN_IF_UNDOCUMENTED NO)
  doxygen_add_docs(doxygen ${PROJECT_SOURCE_DIR}/src)
endif()

if ("${BUILD_TESTING}")
  add_subdirectory(tests)
endif()
