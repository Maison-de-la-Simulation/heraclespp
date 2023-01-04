add_executable(nova++ main.cpp)
target_link_libraries(nova++ PUBLIC Kokkos::kokkos)

set_target_properties(nova++ PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
