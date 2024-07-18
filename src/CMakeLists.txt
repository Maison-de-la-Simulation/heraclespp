find_package(PDI REQUIRED)
find_package(MPI REQUIRED)

add_subdirectory(analysis)
add_subdirectory(gravity)
add_subdirectory(hydro)
add_subdirectory(io)
add_subdirectory(mesh)
add_subdirectory(thermodynamics)
add_subdirectory(units)
add_subdirectory(user_step)
add_subdirectory(utils)

add_library(libnova++ boundary.cpp boundary_distribute.cpp broadcast.cpp initialization_interface.cpp mpi_scope_guard.cpp)
target_include_directories(libnova++ PUBLIC setups/${Novapp_SETUP}/)
target_include_directories(libnova++ PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_features(libnova++ PUBLIC cxx_std_17)

target_link_libraries(libnova++ PUBLIC MPI::MPI_CXX)
target_link_libraries(libnova++ PUBLIC Kokkos::kokkos)
target_link_libraries(libnova++ PUBLIC PDI::PDI_C)
target_link_libraries(libnova++ PUBLIC inih)
target_link_libraries(libnova++ PUBLIC thermodynamics)
target_link_libraries(libnova++ PUBLIC analysis gravity hydro io mesh units user_step utils)

add_executable(nova++ main.cpp)
target_link_libraries(nova++ PUBLIC libnova++)
target_compile_definitions(nova++ PUBLIC NOVAPP_GRAVITY_${Novapp_GRAVITY})
target_compile_definitions(nova++ PUBLIC MY_SETUP="${Novapp_SETUP}")

add_custom_command(TARGET nova++ POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:nova++> ${PROJECT_SOURCE_DIR}/bin/nova++)
