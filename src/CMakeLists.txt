add_subdirectory(thermodynamics)

add_library(libnova++ array_conversion.cpp
                      io.cpp
                      cfl_cond.cpp
                      mpi_scope_guard.cpp
                      grid.cpp
                      buffer.cpp
                      exchange.cpp
                      range.cpp)
target_include_directories(libnova++ INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_features(libnova++ PUBLIC cxx_std_17)

find_package(PDI REQUIRED)
find_package(MPI REQUIRED)

target_link_libraries(libnova++ PUBLIC MPI::MPI_CXX)
target_link_libraries(libnova++ PUBLIC Kokkos::kokkos)
target_link_libraries(libnova++ PUBLIC PDI::PDI_C)
target_link_libraries(libnova++ PUBLIC inih)
target_link_libraries(libnova++ PUBLIC thermodynamics)

# if(NOT NDIM)
#     set(NDIM "1" CACHE STRING "Dimension" FORCE)
# endif()
# if( NOT ((${NDIM} EQUAL 1) OR (${NDIM} EQUAL 2) OR (${NDIM} EQUAL 3)))
#     unset(NDIM CACHE)
#     message(FATAL_ERROR "Please set NDIM = 1(default) or 2 or 3")
# endif()
# configure_file(config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/ndim.hpp @ONLY)


add_executable(nova++ main.cpp)
target_link_libraries(nova++ PUBLIC libnova++)
set_target_properties(nova++ PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# unset(NDIM CACHE)