cmake_minimum_required(VERSION 3.5)
project(inih LANGUAGES C CXX VERSION 1.0)

add_library(${PROJECT_NAME} ini.c INIReader.cpp)
target_include_directories(${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
