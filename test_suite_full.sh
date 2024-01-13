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

BUILD_DIRECTORY="$DIRECTORY/build"

cmake -DBUILD_TESTING=OFF -DNovapp_SETUP=advection_gap -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian -DKokkos_ENABLE_DEPRECATED_CODE_4= -S . -B "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DBUILD_TESTING=OFF -DNovapp_SETUP=advection_gaussian -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian -DKokkos_ENABLE_DEPRECATED_CODE_4= -S . -B "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DBUILD_TESTING=OFF -DNovapp_SETUP=sedov1d -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian -DKokkos_ENABLE_DEPRECATED_CODE_4= -S . -B "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DBUILD_TESTING=OFF -DNovapp_SETUP=shock_tube -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian -DKokkos_ENABLE_DEPRECATED_CODE_4= -S . -B "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DBUILD_TESTING=OFF -DNovapp_SETUP=shock_wall -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian -DKokkos_ENABLE_DEPRECATED_CODE_4= -S . -B "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DBUILD_TESTING=OFF -DNovapp_SETUP=stratified_atm -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian -DKokkos_ENABLE_DEPRECATED_CODE_4= -S . -B "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DBUILD_TESTING=OFF -DNovapp_SETUP=eq_hydro_sphe -DNovapp_NDIM=1 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Point_mass -DNovapp_GEOM=Spherical -DKokkos_ENABLE_DEPRECATED_CODE_4= -S . -B "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DBUILD_TESTING=OFF -DNovapp_SETUP=gresho_vortex -DNovapp_NDIM=2 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian -DKokkos_ENABLE_DEPRECATED_CODE_4= -S . -B "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DBUILD_TESTING=OFF -DNovapp_SETUP=heat_nickel -DNovapp_NDIM=2 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian -DKokkos_ENABLE_DEPRECATED_CODE_4= -S . -B "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DBUILD_TESTING=OFF -DNovapp_SETUP=implosion_test -DNovapp_NDIM=2 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian -DKokkos_ENABLE_DEPRECATED_CODE_4= -S . -B "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DBUILD_TESTING=OFF -DNovapp_SETUP=kelvin_helmholtz -DNovapp_NDIM=2 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian -DKokkos_ENABLE_DEPRECATED_CODE_4= -S . -B "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DBUILD_TESTING=OFF -DNovapp_SETUP=rayleigh_taylor -DNovapp_NDIM=2 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian -DKokkos_ENABLE_DEPRECATED_CODE_4= -S . -B "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DBUILD_TESTING=OFF -DNovapp_SETUP=sedov2d -DNovapp_NDIM=2 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian -DKokkos_ENABLE_DEPRECATED_CODE_4= -S . -B "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DBUILD_TESTING=OFF -DNovapp_SETUP=rayleigh_taylor3d -DNovapp_NDIM=3 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Cartesian -DKokkos_ENABLE_DEPRECATED_CODE_4= -S . -B "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DBUILD_TESTING=OFF -DNovapp_SETUP=shock_tube -DNovapp_NDIM=3 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Spherical -DKokkos_ENABLE_DEPRECATED_CODE_4= -S . -B "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DBUILD_TESTING=OFF -DNovapp_SETUP=sedov3d -DNovapp_NDIM=3 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Spherical -DKokkos_ENABLE_DEPRECATED_CODE_4= -S . -B "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

cmake -DBUILD_TESTING=OFF -DNovapp_SETUP=rayleigh_taylor3d_sph -DNovapp_NDIM=3 -DNovapp_EOS=PerfectGas -DNovapp_GRAVITY=Uniform -DNovapp_GEOM=Spherical -DKokkos_ENABLE_DEPRECATED_CODE_4= -S . -B "$BUILD_DIRECTORY"
cmake --build "$BUILD_DIRECTORY"

rm -rf -- "$DIRECTORY"

trap - EXIT
