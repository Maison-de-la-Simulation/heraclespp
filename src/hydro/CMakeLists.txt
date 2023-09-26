add_library(hydro array_conversion.cpp time_step.cpp)
target_link_libraries(hydro PUBLIC Kokkos::kokkos mesh MPI::MPI_CXX utils)
target_include_directories(hydro INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})