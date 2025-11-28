<!--
SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT
-->

# Building Nova++ on the H100 partition of Jean-Zay

## Environment

```bash
# nova-env.sh
module purge
module load arch/h100
module load gcc/12.2.0
module load cuda/12.4.1
module load openmpi/4.1.5-cuda
module load hdf5/1.12.0-mpi-cuda
module load cmake/3.30.1

. pdi-installation-prefix-path/share/pdi/env.bash
```

Notice that the last line assumes PDI is already installed, see next section.

## How to build PDI

From the nova directory

```bash
cmake \
    -B build-pdi \
    -S vendor/pdi \
    -D BUILD_BENCHMARKING=OFF \
    -D BUILD_DECL_HDF5_PLUGIN=ON \
    -D BUILD_FORTRAN=OFF \
    -D BUILD_HDF5_PARALLEL=ON \
    -D BUILD_DECL_NETCDF_PLUGIN=OFF \
    -D BUILD_MPI_PLUGIN=ON \
    -D BUILD_NETCDF_PARALLEL=OFF \
    -D BUILD_SERIALIZE_PLUGIN=OFF \
    -D BUILD_SET_VALUE_PLUGIN=OFF \
    -D BUILD_SHARED_LIBS=ON \
    -D BUILD_TESTING=OFF \
    -D BUILD_TRACE_PLUGIN=ON \
    -D BUILD_USER_CODE_PLUGIN=OFF \
    -D CMAKE_BUILD_TYPE=Release \
    -D USE_HDF5=SYSTEM \
    -D USE_paraconf=EMBEDDED \
    -D USE_spdlog=EMBEDDED \
    -D USE_yaml=EMBEDDED
cmake --build build-pdi --parallel 4
cmake --install build-pdi --prefix pdi-installation-prefix-path
```

## How to build Nova++

From the top-level nova directory

```bash
cmake \
    -B build \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_COMPILER=$PWD/vendor/kokkos/bin/nvcc_wrapper \
    -D Kokkos_ARCH_HOPPER90=ON \
    -D Kokkos_ENABLE_DEPRECATED_CODE_5=ON \
    -D Kokkos_ENABLE_CUDA=ON \
    -D Kokkos_ENABLE_OPENMP=ON \
    -D Novapp_SETUP=rayleigh_taylor3d \
    -D Novapp_NDIM=3 \
    -D Novapp_EOS=PerfectGas \
    -D Novapp_GRAVITY=Uniform \
    -D Novapp_GEOM=Cartesian
cmake --build build --parallel 4
```
