cmake_minimum_required(VERSION 3.20)

project(Nova++ LANGUAGES CXX)

set(Novapp_NDIM "" CACHE STRING "Choose the dimension, options are: 1;2;3")
set_property(CACHE Novapp_NDIM PROPERTY STRINGS "1;2;3")
if("${Novapp_NDIM}" STREQUAL "")
  message(FATAL_ERROR "Novapp_NDIM is not defined")
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
