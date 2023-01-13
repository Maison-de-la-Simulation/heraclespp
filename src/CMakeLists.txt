add_executable(nova++ main.cpp init.cpp fonc.cpp slope.cpp flux.cpp conv.cpp solver.cpp io.cpp)

find_package(PDI REQUIRED)
target_link_libraries(nova++ PUBLIC Kokkos::kokkos)
target_link_libraries(nova++ PUBLIC PDI::PDI_C)

set_target_properties(nova++ PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
