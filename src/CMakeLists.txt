add_subdirectory(thermodynamics)

add_executable(nova++ main.cpp 
                      array_conversion.cpp 
                      io.cpp 
                      cfl_cond.cpp 
                      mpi_scope_guard.cpp 
                      grid.cpp
                      buffer.cpp
                      range.cpp)

target_compile_features(nova++ PUBLIC cxx_std_17)

find_package(PDI REQUIRED)
find_package(MPI REQUIRED)

target_link_libraries(nova++ PUBLIC MPI::MPI_CXX)
target_link_libraries(nova++ PUBLIC Kokkos::kokkos)
target_link_libraries(nova++ PUBLIC PDI::PDI_C)
target_link_libraries(nova++ PUBLIC inih)
target_link_libraries(nova++ PUBLIC thermodynamics)

set_target_properties(nova++ PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
