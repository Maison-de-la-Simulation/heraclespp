#! /bin/bash

export CMAKE_BUILD_PARALLEL_LEVEL=4
export CMAKE_BUILD_TYPE="Release"

DIRECTORY=$(mktemp -d)
if [ $? -ne 0 ]; then
    echo "$0: Can't create temporary directory, exiting."
    exit 1
fi

trap 'rm -rf -- "$DIRECTORY"' EXIT
set -xe

# Install Kokkos
cmake -B "$DIRECTORY/build-kokkos" -S vendor/kokkos -DBUILD_TESTING=OFF -DKokkos_ENABLE_DEPRECATED_CODE_4=OFF
cmake --build "$DIRECTORY/build-kokkos"
cmake --install "$DIRECTORY/build-kokkos" --prefix "$DIRECTORY/install-kokkos"
export Kokkos_ROOT="$DIRECTORY/install-kokkos"

# Install GTest
cmake -B "$DIRECTORY/build-gtest" -S vendor/googletest
cmake --build "$DIRECTORY/build-gtest"
cmake --install "$DIRECTORY/build-gtest" --prefix "$DIRECTORY/install-gtest"
export GTest_ROOT="$DIRECTORY/install-gtest"

export CXXFLAGS="-Werror=all -Werror=extra -Werror=pedantic -pedantic-errors"
export BUILD_DIRECTORY="$DIRECTORY/build-novapp"

# Warm up, first configuration with shared options
cmake -B "$BUILD_DIRECTORY" -DBUILD_TESTING=OFF -DNovapp_GTest_DEPENDENCY_POLICY=INSTALLED -DNovapp_Kokkos_DEPENDENCY_POLICY=INSTALLED -DNovapp_SETUP=advection_gap -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian

# Configure and build all setups (alphabetical order)
cmake "$BUILD_DIRECTORY" -DNovapp_SETUP=advection_gap -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian
cmake --build "$BUILD_DIRECTORY"

cmake "$BUILD_DIRECTORY" -DNovapp_SETUP=advection_sinus -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian
cmake --build "$BUILD_DIRECTORY"

cmake "$BUILD_DIRECTORY" -DNovapp_SETUP=eq_hydro_sphe -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Point_mass -DNovapp_GEOM=Spherical
cmake --build "$BUILD_DIRECTORY"

cmake "$BUILD_DIRECTORY" -DNovapp_SETUP=gresho_vortex -DNovapp_NDIM=2 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian
cmake --build "$BUILD_DIRECTORY"

cmake "$BUILD_DIRECTORY" -DNovapp_SETUP=heat_nickel -DNovapp_NDIM=2 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian
cmake --build "$BUILD_DIRECTORY"

cmake "$BUILD_DIRECTORY" -DNovapp_SETUP=implosion_test -DNovapp_NDIM=2 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian
cmake --build "$BUILD_DIRECTORY"

cmake "$BUILD_DIRECTORY" -DNovapp_SETUP=kelvin_helmholtz -DNovapp_NDIM=2 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian
cmake --build "$BUILD_DIRECTORY"

cmake "$BUILD_DIRECTORY" -DNovapp_SETUP=rayleigh_taylor -DNovapp_NDIM=2 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian
cmake --build "$BUILD_DIRECTORY"

cmake "$BUILD_DIRECTORY" -DNovapp_SETUP=rayleigh_taylor3d -DNovapp_NDIM=3 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian
cmake --build "$BUILD_DIRECTORY"

cmake "$BUILD_DIRECTORY" -DNovapp_SETUP=rayleigh_taylor3d_sph -DNovapp_NDIM=3 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Spherical
cmake --build "$BUILD_DIRECTORY"

cmake "$BUILD_DIRECTORY" -DNovapp_SETUP=sedov1d -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian
cmake --build "$BUILD_DIRECTORY"

cmake "$BUILD_DIRECTORY" -DNovapp_SETUP=sedov2d -DNovapp_NDIM=2 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian
cmake --build "$BUILD_DIRECTORY"

cmake "$BUILD_DIRECTORY" -DNovapp_SETUP=sedov2d -DNovapp_NDIM=2 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Spherical
cmake --build "$BUILD_DIRECTORY"

cmake "$BUILD_DIRECTORY" -DNovapp_SETUP=sedov3d -DNovapp_NDIM=3 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Spherical
cmake --build "$BUILD_DIRECTORY"

cmake "$BUILD_DIRECTORY" -DNovapp_SETUP=shock_tube -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian
cmake --build "$BUILD_DIRECTORY"

cmake "$BUILD_DIRECTORY" -DNovapp_SETUP=shock_tube -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Spherical
cmake --build "$BUILD_DIRECTORY"

cmake "$BUILD_DIRECTORY" -DNovapp_SETUP=shock_tube -DNovapp_NDIM=2 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Spherical
cmake --build "$BUILD_DIRECTORY"

cmake "$BUILD_DIRECTORY" -DNovapp_SETUP=shock_tube -DNovapp_NDIM=3 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Spherical
cmake --build "$BUILD_DIRECTORY"

cmake "$BUILD_DIRECTORY" -DNovapp_SETUP=shock_wall -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian
cmake --build "$BUILD_DIRECTORY"

cmake "$BUILD_DIRECTORY" -DNovapp_SETUP=stratified_atm -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian
cmake --build "$BUILD_DIRECTORY"

cmake "$BUILD_DIRECTORY" -DNovapp_SETUP=v1d -DNovapp_NDIM=1 -DNovapp_EOS=Gas+Radiation -DNovapp_GRAVITY=Point_mass -DNovapp_GEOM=Spherical
cmake --build "$BUILD_DIRECTORY"

cmake "$BUILD_DIRECTORY" -DNovapp_SETUP=v1d -DNovapp_NDIM=2 -DNovapp_EOS=Gas+Radiation -DNovapp_GRAVITY=Point_mass -DNovapp_GEOM=Spherical
cmake --build "$BUILD_DIRECTORY"

cmake "$BUILD_DIRECTORY" -DNovapp_SETUP=v1d -DNovapp_NDIM=3 -DNovapp_EOS=Gas+Radiation -DNovapp_GRAVITY=Point_mass -DNovapp_GEOM=Spherical
cmake --build "$BUILD_DIRECTORY"

rm -rf -- "$DIRECTORY"

trap - EXIT
