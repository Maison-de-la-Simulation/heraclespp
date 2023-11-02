#! /bin/bash
export CMAKE_BUILD_PARALLEL_LEVEL=4
export CMAKE_BUILD_TYPE="Release"
export CXXFLAGS="-Wall -Wextra"

DIRECTORY=$(mktemp -d)
if [ $? -ne 0 ]; then
    echo "$0: Can't create temporary directory, exiting."
    exit 1
fi

trap 'rm -rf -- "$DIRECTORY"' EXIT
set -xe

SETUP_NAME="advection_gap"
cmake -DBUILD_TESTING=OFF -DNovapp_SETUP="$SETUP_NAME" -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian -S . -B "$DIRECTORY/build_$SETUP_NAME"
cmake --build "$DIRECTORY/build_$SETUP_NAME"

SETUP_NAME="advection_gaussian"
cmake -DBUILD_TESTING=OFF -DNovapp_SETUP="$SETUP_NAME" -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian -S . -B "$DIRECTORY/build_$SETUP_NAME"
cmake --build "$DIRECTORY/build_$SETUP_NAME"

SETUP_NAME="eq_hydro_sphe"
cmake -DBUILD_TESTING=OFF -DNovapp_SETUP="$SETUP_NAME" -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Point_mass -DNovapp_GEOM=Spherical -S . -B "$DIRECTORY/build_$SETUP_NAME"
cmake --build "$DIRECTORY/build_$SETUP_NAME"

SETUP_NAME="gresho_vortex"
cmake -DBUILD_TESTING=OFF -DNovapp_SETUP="$SETUP_NAME" -DNovapp_NDIM=2 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian -S . -B "$DIRECTORY/build_$SETUP_NAME"
cmake --build "$DIRECTORY/build_$SETUP_NAME"

SETUP_NAME="heat_nickel"
cmake -DBUILD_TESTING=OFF -DNovapp_SETUP="$SETUP_NAME" -DNovapp_NDIM=2 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian -S . -B "$DIRECTORY/build_$SETUP_NAME"
cmake --build "$DIRECTORY/build_$SETUP_NAME"

SETUP_NAME="implosion_test"
cmake -DBUILD_TESTING=OFF -DNovapp_SETUP="$SETUP_NAME" -DNovapp_NDIM=2 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian -S . -B "$DIRECTORY/build_$SETUP_NAME"
cmake --build "$DIRECTORY/build_$SETUP_NAME"

SETUP_NAME="kelvin_helmholtz"
cmake -DBUILD_TESTING=OFF -DNovapp_SETUP="$SETUP_NAME" -DNovapp_NDIM=2 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian -S . -B "$DIRECTORY/build_$SETUP_NAME"
cmake --build "$DIRECTORY/build_$SETUP_NAME"

SETUP_NAME="rayleigh_taylor"
cmake -DBUILD_TESTING=OFF -DNovapp_SETUP="$SETUP_NAME" -DNovapp_NDIM=2 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian -S . -B "$DIRECTORY/build_$SETUP_NAME"
cmake --build "$DIRECTORY/build_$SETUP_NAME"

SETUP_NAME="rayleigh_taylor3d"
cmake -DBUILD_TESTING=OFF -DNovapp_SETUP="$SETUP_NAME" -DNovapp_NDIM=3 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian -S . -B "$DIRECTORY/build_$SETUP_NAME"
cmake --build "$DIRECTORY/build_$SETUP_NAME"

SETUP_NAME="sedov1d"
cmake -DBUILD_TESTING=OFF -DNovapp_SETUP="$SETUP_NAME" -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian -S . -B "$DIRECTORY/build_$SETUP_NAME"
cmake --build "$DIRECTORY/build_$SETUP_NAME"

SETUP_NAME="sedov2d"
cmake -DBUILD_TESTING=OFF -DNovapp_SETUP="$SETUP_NAME" -DNovapp_NDIM=2 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian -S . -B "$DIRECTORY/build_$SETUP_NAME"
cmake --build "$DIRECTORY/build_$SETUP_NAME"

SETUP_NAME="shock_tube"
cmake -DBUILD_TESTING=OFF -DNovapp_SETUP="$SETUP_NAME" -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian -S . -B "$DIRECTORY/build_$SETUP_NAME"
cmake --build "$DIRECTORY/build_$SETUP_NAME"

SETUP_NAME="shock_wall"
cmake -DBUILD_TESTING=OFF -DNovapp_SETUP="$SETUP_NAME" -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian -S . -B "$DIRECTORY/build_$SETUP_NAME"
cmake --build "$DIRECTORY/build_$SETUP_NAME"

SETUP_NAME="stratified_atm"
cmake -DBUILD_TESTING=OFF -DNovapp_SETUP="$SETUP_NAME" -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian -S . -B "$DIRECTORY/build_$SETUP_NAME"
cmake --build "$DIRECTORY/build_$SETUP_NAME"

rm -rf -- "$DIRECTORY"

trap - EXIT
