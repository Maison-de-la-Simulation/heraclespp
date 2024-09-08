add_library(moving_grid moving_grid.cpp)
target_include_directories(moving_grid INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features( moving_grid PUBLIC cxx_std_17)

target_link_libraries(moving_grid PUBLIC Kokkos::kokkos)
target_link_libraries(moving_grid PUBLIC utils mesh)
