add_executable(nova++ main.cpp init.cpp fonc.cpp flux.cpp conv.cpp solver.cpp io.cpp boundary.cpp mpi_fonc.cpp)
target_compile_features(nova++ PUBLIC cxx_std_17)

find_package(PDI REQUIRED)
target_link_libraries(nova++ PUBLIC Kokkos::kokkos)
target_link_libraries(nova++ PUBLIC PDI::PDI_C)
target_link_libraries(nova++ PUBLIC inih)


set_target_properties(nova++ PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
