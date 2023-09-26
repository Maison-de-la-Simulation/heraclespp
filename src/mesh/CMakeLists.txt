add_library(mesh grid.cpp range.cpp)
target_link_libraries(mesh PUBLIC Kokkos::kokkos MPI::MPI_CXX utils)
target_include_directories(mesh INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})