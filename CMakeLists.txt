cmake_minimum_required(VERSION 3.20)

project(nova++ LANGUAGES CXX)

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
