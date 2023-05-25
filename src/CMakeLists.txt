add_subdirectory(thermodynamics)

add_library(libnova++ array_conversion.cpp
                      io_config.yaml.cpp
                      io.cpp
                      cfl_cond.cpp
                      mpi_scope_guard.cpp
                      grid.cpp
                      boundary.cpp
                      range.cpp)
target_include_directories(libnova++ PUBLIC setups/${Novapp_SETUP}/)
configure_file(ndim.hpp.in ndim.hpp NO_SOURCE_PERMISSIONS @ONLY)
target_include_directories(libnova++ PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(libnova++ PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

target_compile_features(libnova++ PUBLIC cxx_std_17)

find_package(PDI REQUIRED)
find_package(MPI REQUIRED)

target_link_libraries(libnova++ PUBLIC MPI::MPI_CXX)
target_link_libraries(libnova++ PUBLIC Kokkos::kokkos)
target_link_libraries(libnova++ PUBLIC PDI::PDI_C)
target_link_libraries(libnova++ PUBLIC inih)
target_link_libraries(libnova++ PUBLIC thermodynamics)

add_executable(nova++ main.cpp)
target_link_libraries(nova++ PUBLIC libnova++)
set_target_properties(nova++ PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
