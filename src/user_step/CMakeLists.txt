add_library(user_step user_step.cpp user_step_factory.cpp)
target_include_directories(user_step INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(user_step
    PRIVATE Kokkos::kokkos mesh units
    PUBLIC utils)
