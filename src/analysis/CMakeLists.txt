add_library(analysis integration.cpp internal_energy.cpp)
target_link_libraries(analysis PUBLIC hydro Kokkos::kokkos mesh MPI::MPI_CXX utils)
target_include_directories(analysis PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
