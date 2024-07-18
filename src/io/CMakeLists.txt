find_package(HDF5 REQUIRED COMPONENTS C)

add_library(io io.cpp config.yaml.cpp)
target_link_libraries(io PUBLIC mesh MPI::MPI_CXX PDI::PDI_C utils hdf5::hdf5)
target_include_directories(io INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
