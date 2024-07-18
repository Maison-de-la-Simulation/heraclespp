# Building Nova++ on the A100 partition of Ruche

## Environment

```bash
# nova-env.sh
module purge
module load cmake/3.21.4/gcc-11.2.0
module load gcc/11.2.0/gcc-4.8.5
module load cuda/11.8.0/gcc-11.2.0
module load openmpi/4.1.1/gcc-11.2.0
module load hdf5/1.10.7/gcc-11.2.0-openmpi
module load paraconf/1.0.0/gcc-11.2.0 libyaml/0.2.5/gcc-11.2.0
module load pdi/1.6.0/gcc-11.2.0
```

## How to build Nova++

From the top-level nova directory

```bash
cmake \
    -B build \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_COMPILER=$PWD/vendor/kokkos/bin/nvcc_wrapper \
    -D Kokkos_ARCH_AMPERE80=ON \
    -D Kokkos_ARCH_ICX=ON \
    -D Kokkos_ENABLE_DEPRECATED_CODE_4=OFF \
    -D Kokkos_ENABLE_CUDA=ON \
    -D Kokkos_ENABLE_OPENMP=ON \
    -D Novapp_SETUP=sedov1d \
    -D Novapp_NDIM=1 \
    -D Novapp_EOS=PerfectGas \
    -D Novapp_GRAVITY=Uniform \
    -D Novapp_GEOM=Cartesian
cmake --build build --parallel 4
```
