add_library(
    mesh
    geometry.cpp
    geometry_factory.cpp
    grid.cpp
    grid_factory.cpp
    grid_type.cpp
    range.cpp
)
target_include_directories(mesh INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mesh PUBLIC Kokkos::kokkos MPI::MPI_CXX utils)
