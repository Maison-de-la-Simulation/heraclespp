add_library(hydro INTERFACE)
target_include_directories(hydro INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(hydro INTERFACE Kokkos::kokkos mesh MPI::MPI_CXX utils)
