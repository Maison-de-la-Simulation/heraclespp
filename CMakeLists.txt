cmake_minimum_required(VERSION 3.20)

project(nova++ LANGUAGES CXX)

add_subdirectory(vendor/kokkos)

add_subdirectory(src)

find_package(Doxygen)
if (Doxygen_FOUND)
  doxygen_add_docs(doxygen ${PROJECT_SOURCE_DIR}/src)
endif()
