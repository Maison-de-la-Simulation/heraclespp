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

export CXXFLAGS="-Werror=all -Werror=extra -Werror=pedantic -pedantic-errors"
export BUILD_DIRECTORY="$DIRECTORY/build"

# Warm up, first configuration with shared options
cmake -DBUILD_TESTING=OFF -DNovapp_GTest_DEPENDENCY_POLICY=INSTALLED -DNovapp_inih_DEPENDENCY_POLICY=INSTALLED -DNovapp_Kokkos_DEPENDENCY_POLICY=INSTALLED -DNovapp_SETUP=advection_gap -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian -B "$BUILD_DIRECTORY"

# Configure and build all setups (alphabetical order)
cmake -DNovapp_SETUP=advection_gap -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DNovapp_SETUP=advection_sinus -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DNovapp_SETUP=eq_hydro_sphe -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Point_mass -DNovapp_GEOM=Spherical "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DNovapp_SETUP=gresho_vortex -DNovapp_NDIM=2 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DNovapp_SETUP=heat_nickel -DNovapp_NDIM=2 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DNovapp_SETUP=implosion_test -DNovapp_NDIM=2 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DNovapp_SETUP=kelvin_helmholtz -DNovapp_NDIM=2 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DNovapp_SETUP=rayleigh_taylor -DNovapp_NDIM=2 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DNovapp_SETUP=rayleigh_taylor3d -DNovapp_NDIM=3 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DNovapp_SETUP=rayleigh_taylor3d_sph -DNovapp_NDIM=3 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Spherical "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DNovapp_SETUP=sedov1d -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DNovapp_SETUP=sedov2d -DNovapp_NDIM=2 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DNovapp_SETUP=sedov2d -DNovapp_NDIM=2 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Spherical "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DNovapp_SETUP=sedov3d -DNovapp_NDIM=3 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Spherical "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DNovapp_SETUP=shock_tube -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DNovapp_SETUP=shock_tube -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Spherical "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DNovapp_SETUP=shock_tube -DNovapp_NDIM=2 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Spherical "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DNovapp_SETUP=shock_tube -DNovapp_NDIM=3 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Spherical "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DNovapp_SETUP=shock_wall -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DNovapp_SETUP=stratified_atm -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DNovapp_SETUP=v1d -DNovapp_NDIM=1 -DNovapp_EOS=Gas+Radiation -DNovapp_GRAVITY=Point_mass -DNovapp_GEOM=Spherical "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DNovapp_SETUP=v1d -DNovapp_NDIM=2 -DNovapp_EOS=Gas+Radiation -DNovapp_GRAVITY=Point_mass -DNovapp_GEOM=Spherical "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DNovapp_SETUP=v1d -DNovapp_NDIM=3 -DNovapp_EOS=Gas+Radiation -DNovapp_GRAVITY=Point_mass -DNovapp_GEOM=Spherical "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

rm -rf -- "$DIRECTORY"

trap - EXIT
