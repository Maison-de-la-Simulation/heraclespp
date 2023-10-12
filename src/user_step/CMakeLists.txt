add_library(user_step INTERFACE)

target_link_libraries(user_step INTERFACE Kokkos::kokkos units utils)
target_include_directories(user_step INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
