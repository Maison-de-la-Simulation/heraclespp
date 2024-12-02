cmake_minimum_required(VERSION 3.5)
project(inih LANGUAGES C CXX VERSION 1.0)

add_library(${PROJECT_NAME} ini.c INIReader.cpp)
target_include_directories(${PROJECT_NAME}
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    INTERFACE $<INSTALL_INTERFACE:include>)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

include(CMakePackageConfigHelpers)

install(
    TARGETS inih
    EXPORT inihTargets)

install(
    EXPORT inihTargets
    DESTINATION lib/cmake/inih)

install(
    FILES ini.h INIReader.hpp
    DESTINATION include/inih)

configure_package_config_file(cmake/inihConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/inihConfig.cmake
    INSTALL_DESTINATION lib/cmake/inih)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/inihConfig.cmake
    DESTINATION lib/cmake/inih)
