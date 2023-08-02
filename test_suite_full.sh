#! /bin/bash
JOBS=4
DIRECTORY=tmp

set -xe

mkdir "$DIRECTORY"

SETUP_NAME="advection_gap"
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DNovapp_SETUP="$SETUP_NAME" -DNovapp_NDIM=1 -DNovapp_EOS=PERFECT -DNovapp_GRAVITY=UNIFORM -DNovapp_GEOM=CARTESIAN -S . -B "$DIRECTORY/build_$SETUP_NAME"
cmake --build "$DIRECTORY/build_$SETUP_NAME" --parallel "$JOBS"

SETUP_NAME="advection_gaussian"
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DNovapp_SETUP="$SETUP_NAME" -DNovapp_NDIM=1 -DNovapp_EOS=PERFECT -DNovapp_GRAVITY=UNIFORM -DNovapp_GEOM=CARTESIAN -S . -B "$DIRECTORY/build_$SETUP_NAME"
cmake --build "$DIRECTORY/build_$SETUP_NAME" --parallel "$JOBS"

SETUP_NAME="eq_hydro_sphe"
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DNovapp_SETUP="$SETUP_NAME" -DNovapp_NDIM=1 -DNovapp_EOS=PERFECT -DNovapp_GRAVITY=POINT_MASS -DNovapp_GEOM=SPHERICAL -S . -B "$DIRECTORY/build_$SETUP_NAME"
cmake --build "$DIRECTORY/build_$SETUP_NAME" --parallel "$JOBS"

SETUP_NAME="gresho_vortex"
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DNovapp_SETUP="$SETUP_NAME" -DNovapp_NDIM=2 -DNovapp_EOS=PERFECT -DNovapp_GRAVITY=UNIFORM -DNovapp_GEOM=CARTESIAN -S . -B "$DIRECTORY/build_$SETUP_NAME"
cmake --build "$DIRECTORY/build_$SETUP_NAME" --parallel "$JOBS"

SETUP_NAME="implosion_test"
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DNovapp_SETUP="$SETUP_NAME" -DNovapp_NDIM=2 -DNovapp_EOS=PERFECT -DNovapp_GRAVITY=UNIFORM -DNovapp_GEOM=CARTESIAN -S . -B "$DIRECTORY/build_$SETUP_NAME"
cmake --build "$DIRECTORY/build_$SETUP_NAME" --parallel "$JOBS"

SETUP_NAME="kelvin_helmholtz"
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DNovapp_SETUP="$SETUP_NAME" -DNovapp_NDIM=2 -DNovapp_EOS=PERFECT -DNovapp_GRAVITY=UNIFORM -DNovapp_GEOM=CARTESIAN -S . -B "$DIRECTORY/build_$SETUP_NAME"
cmake --build "$DIRECTORY/build_$SETUP_NAME" --parallel "$JOBS"

SETUP_NAME="rayleigh_taylor"
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DNovapp_SETUP="$SETUP_NAME" -DNovapp_NDIM=2 -DNovapp_EOS=PERFECT -DNovapp_GRAVITY=UNIFORM -DNovapp_GEOM=CARTESIAN -S . -B "$DIRECTORY/build_$SETUP_NAME"
cmake --build "$DIRECTORY/build_$SETUP_NAME" --parallel "$JOBS"

SETUP_NAME="rayleigh_taylor3d"
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DNovapp_SETUP="$SETUP_NAME" -DNovapp_NDIM=3 -DNovapp_EOS=PERFECT -DNovapp_GRAVITY=UNIFORM -DNovapp_GEOM=CARTESIAN -S . -B "$DIRECTORY/build_$SETUP_NAME"
cmake --build "$DIRECTORY/build_$SETUP_NAME" --parallel "$JOBS"

SETUP_NAME="sedov1d"
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DNovapp_SETUP="$SETUP_NAME" -DNovapp_NDIM=1 -DNovapp_EOS=PERFECT -DNovapp_GRAVITY=UNIFORM -DNovapp_GEOM=CARTESIAN -S . -B "$DIRECTORY/build_$SETUP_NAME"
cmake --build "$DIRECTORY/build_$SETUP_NAME" --parallel "$JOBS"

SETUP_NAME="sedov2d"
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DNovapp_SETUP="$SETUP_NAME" -DNovapp_NDIM=2 -DNovapp_EOS=PERFECT -DNovapp_GRAVITY=UNIFORM -DNovapp_GEOM=CARTESIAN -S . -B "$DIRECTORY/build_$SETUP_NAME"
cmake --build "$DIRECTORY/build_$SETUP_NAME" --parallel "$JOBS"

SETUP_NAME="shock_tube"
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DNovapp_SETUP="$SETUP_NAME" -DNovapp_NDIM=1 -DNovapp_EOS=PERFECT -DNovapp_GRAVITY=UNIFORM -DNovapp_GEOM=CARTESIAN -S . -B "$DIRECTORY/build_$SETUP_NAME"
cmake --build "$DIRECTORY/build_$SETUP_NAME" --parallel "$JOBS"

SETUP_NAME="shock_wall"
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DNovapp_SETUP="$SETUP_NAME" -DNovapp_NDIM=1 -DNovapp_EOS=PERFECT -DNovapp_GRAVITY=UNIFORM -DNovapp_GEOM=CARTESIAN -S . -B "$DIRECTORY/build_$SETUP_NAME"
cmake --build "$DIRECTORY/build_$SETUP_NAME" --parallel "$JOBS"

SETUP_NAME="stratified_atm"
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DNovapp_SETUP="$SETUP_NAME" -DNovapp_NDIM=1 -DNovapp_EOS=PERFECT -DNovapp_GRAVITY=UNIFORM -DNovapp_GEOM=CARTESIAN -S . -B "$DIRECTORY/build_$SETUP_NAME"
cmake --build "$DIRECTORY/build_$SETUP_NAME" --parallel "$JOBS"

rm -rf "$DIRECTORY"
