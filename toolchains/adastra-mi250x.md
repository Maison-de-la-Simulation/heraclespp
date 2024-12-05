# Building Nova++ on the MI250X partition of Adastra

## Environment

```bash
# nova-env.sh
module purge
module load cpe/24.07
module load craype-x86-trento craype-accel-amd-gfx90a
module load PrgEnv-amd
module load cray-hdf5-parallel

export MPICH_GPU_SUPPORT_ENABLED=1
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
    -D CMAKE_CXX_COMPILER=hipcc \
    -D CMAKE_CXX_FLAGS="-isystem${CRAY_HDF5_PARALLEL_PREFIX}/include" \
    -D CMAKE_EXE_LINKER_FLAGS="-L${CRAY_HDF5_PARALLEL_PREFIX}/lib -lhdf5_hl_parallel -lhdf5_parallel" \
    -D Kokkos_ARCH_AMD_GFX90A=ON \
    -D Kokkos_ARCH_ZEN3=ON \
    -D Kokkos_ENABLE_DEPRECATED_CODE_4=OFF \
    -D Kokkos_ENABLE_HIP=ON \
    -D Kokkos_ENABLE_OPENMP=ON \
    -D Novapp_SETUP=sedov1d \
    -D Novapp_NDIM=1 \
    -D Novapp_EOS=PerfectGas \
    -D Novapp_GRAVITY=Uniform \
    -D Novapp_GEOM=Cartesian
cmake --build build --parallel 4
```
