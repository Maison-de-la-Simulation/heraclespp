add_library(utils nova_params.cpp)
target_link_libraries(utils PUBLIC inih)
target_link_libraries(utils INTERFACE Kokkos::kokkos thermodynamics)
target_include_directories(utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(utils INTERFACE ${CMAKE_CURRENT_BINARY_DIR})

configure_file(eos.hpp.in eos.hpp NO_SOURCE_PERMISSIONS @ONLY)
configure_file(geom.hpp.in geom.hpp NO_SOURCE_PERMISSIONS @ONLY)
configure_file(ndim.hpp.in ndim.hpp NO_SOURCE_PERMISSIONS @ONLY)
