find_package(PDI REQUIRED)
find_package(MPI REQUIRED)

add_subdirectory(analysis)
add_subdirectory(gravity)
add_subdirectory(hydro)
add_subdirectory(io)
add_subdirectory(mesh)
add_subdirectory(thermodynamics)
add_subdirectory(units)
add_subdirectory(utils)

add_library(libnova++ mpi_scope_guard.cpp
                      boundary_distribute.cpp)
target_include_directories(libnova++ PUBLIC setups/${Novapp_SETUP}/)
target_include_directories(libnova++ PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_features(libnova++ PUBLIC cxx_std_17)

target_link_libraries(libnova++ PUBLIC MPI::MPI_CXX)
target_link_libraries(libnova++ PUBLIC Kokkos::kokkos)
target_link_libraries(libnova++ PUBLIC PDI::PDI_C)
target_link_libraries(libnova++ PUBLIC inih)
target_link_libraries(libnova++ PUBLIC thermodynamics)
target_link_libraries(libnova++ PUBLIC analysis gravity hydro io mesh units utils)

add_executable(nova++ main.cpp)
target_link_libraries(nova++ PUBLIC libnova++)
set_target_properties(nova++ PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

target_compile_definitions(nova++ PUBLIC ${Novapp_GRAVITY})

target_compile_definitions(nova++ PUBLIC MY_SETUP="${Novapp_SETUP}")
