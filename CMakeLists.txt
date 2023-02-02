cmake_minimum_required(VERSION 3.20)

project(nova++ LANGUAGES CXX)

add_subdirectory(vendor/inih)
add_subdirectory(vendor/kokkos)

add_subdirectory(src)

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
target_link_libraries(nova++ PUBLIC MPI::MPI_CXX)

find_package(Doxygen)
if (Doxygen_FOUND)
  doxygen_add_docs(doxygen ${PROJECT_SOURCE_DIR}/src)
endif()
