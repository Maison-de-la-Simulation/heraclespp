add_subdirectory(thermodynamics)

add_executable(nova++ main.cpp init.cpp array_conversion.cpp io.cpp domain_dec.cpp cfl_cond.cpp)
target_compile_features(nova++ PUBLIC cxx_std_17)

find_package(PDI REQUIRED)
target_link_libraries(nova++ PUBLIC Kokkos::kokkos)
target_link_libraries(nova++ PUBLIC PDI::PDI_C)
target_link_libraries(nova++ PUBLIC inih)
target_link_libraries(nova++ PUBLIC thermodynamics)

set_target_properties(nova++ PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
