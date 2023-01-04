cmake_minimum_required(VERSION 3.20)

project(nova++ LANGUAGES CXX)

option(Nova++_BUILD_DOCUMENTATION "Build documentation" OFF)

add_subdirectory(vendor/kokkos)

add_subdirectory(src)

if ("${Nova++_BUILD_DOCUMENTATION}")
  find_package(Doxygen REQUIRED)
  doxygen_add_docs(doxygen ${PROJECT_SOURCE_DIR}/src)
endif ()
