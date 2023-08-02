add_library(units INTERFACE)

target_link_libraries(units INTERFACE Kokkos::kokkos)
target_include_directories(units INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
