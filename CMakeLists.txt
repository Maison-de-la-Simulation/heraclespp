cmake_minimum_required(VERSION 3.20)

project(Nova++ LANGUAGES CXX)

set(DIMS "1;2;3")
set(Novapp_NDIM "" CACHE STRING "Choose the dimension, options are ${DIMS}")
set_property(CACHE Novapp_NDIM PROPERTY STRINGS "${DIMS}")
if(NOT ("${Novapp_NDIM}"  IN_LIST DIMS))
  message(FATAL_ERROR "Novapp_NDIM is not defined, available dimensions are ${DIMS}")
endif()



set(Novapp_SETUP "" CACHE STRING "Choose the setup")
file(GLOB sub-dirs-name  RELATIVE ${PROJECT_SOURCE_DIR}/src/setups ${PROJECT_SOURCE_DIR}/src/setups/*)
set_property(CACHE Novapp_SETUP PROPERTY STRINGS "${sub-dirs-name}")

if (NOT("${Novapp_SETUP}" IN_LIST sub-dirs-name))
  message(FATAL_ERROR "Novapp_SETUP is not defined, available setups are: ${sub-dirs-name}")
endif()


if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. ")
endif()

include(CTest)

add_subdirectory(vendor/googletest)
add_subdirectory(vendor/inih)
add_subdirectory(vendor/kokkos)

add_subdirectory(src)

find_package(Doxygen)
if (Doxygen_FOUND)
  doxygen_add_docs(doxygen ${PROJECT_SOURCE_DIR}/src)
endif()

if ("${BUILD_TESTING}")
  add_subdirectory(tests)
endif()
