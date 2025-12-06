# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

add_library(hydro)
target_compile_features(hydro PUBLIC cxx_std_20)
target_link_libraries(
    hydro
    PUBLIC Kokkos::kokkos mesh utils
    INTERFACE MPI::MPI_CXX
)
target_sources(
    hydro
    PRIVATE
        array_conversion.cpp
        concepts.cpp
        euler_equations.cpp
        extrapolation_reconstruction.cpp
        face_reconstruction.cpp
        godunov_scheme.cpp
        hydro_reconstruction.cpp
        limited_linear_reconstruction.cpp
        pressure_fix.cpp
        riemann_solver.cpp
        slope_limiters.cpp
        source_terms.cpp
        time_step.cpp
    INTERFACE
        FILE_SET HEADERS
            FILES
                array_conversion.hpp
                concepts.hpp
                euler_equations.hpp
                extrapolation_reconstruction.hpp
                face_reconstruction.hpp
                godunov_scheme.hpp
                hydro_reconstruction.hpp
                limited_linear_reconstruction.hpp
                pressure_fix.hpp
                riemann_solver.hpp
                slope_limiters.hpp
                source_terms.hpp
                time_step.hpp
)
