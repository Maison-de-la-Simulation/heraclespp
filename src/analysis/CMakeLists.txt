add_library(analysis integration.cpp)
target_link_libraries(analysis PUBLIC Kokkos::kokkos mesh utils)
target_include_directories(analysis PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})