add_library(analysis INTERFACE)
target_link_libraries(analysis INTERFACE Kokkos::kokkos mesh utils)
target_include_directories(analysis INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})