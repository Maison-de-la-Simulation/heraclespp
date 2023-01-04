cmake_minimum_required(VERSION 3.20)

project(nova++ LANGUAGES CXX)

option(Nova++_BUILD_DOCUMENTATION "Build documentation" OFF)

add_subdirectory(vendor/kokkos)

set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries rather than static ones" FORCE)
set(BUILD_FORTRAN OFF CACHE BOOL "Build with Fortran support" FORCE)
set(BUILD_BENCHMARKING OFF CACHE BOOL "Build PDI benchmarks" FORCE)
set(BUILD_SET_VALUE_PLUGIN CACHE BOOL "Build Set_value plug-in" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Build tests" FORCE)
set(BUILD_DECL_NETCDF_PLUGIN OFF CACHE BOOL "Build Decl'NetCDF plug-in" FORCE)
set(BUILD_NETCDF_PARALLEL OFF CACHE BOOL "Build Decl'NetCDF in parallel mode" FORCE)
add_subdirectory(vendor/pdi)

add_subdirectory(src)
