cmake_minimum_required(VERSION 3.25)
project(inih LANGUAGES C CXX VERSION 1.0)

add_library(inih)
target_sources(
    inih
    PRIVATE src/ini.c src/INIReader.cpp
    PUBLIC
        FILE_SET HEADERS
            BASE_DIRS include
            FILES include/inih/ini.h include/inih/INIReader.hpp
)
target_compile_features(inih PUBLIC cxx_std_11)

include(CMakePackageConfigHelpers)

install(TARGETS inih EXPORT inihTargets FILE_SET HEADERS)

install(EXPORT inihTargets DESTINATION lib/cmake/inih)

configure_package_config_file(
    cmake/inihConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/inihConfig.cmake
    INSTALL_DESTINATION lib/cmake/inih
)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/inihConfig.cmake
    DESTINATION lib/cmake/inih
)
