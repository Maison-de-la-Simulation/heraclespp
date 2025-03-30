# HERACLES++

## How to get sources

To get the full source of HERACLES++ and external libraries:

```bash
git clone --recurse-submodules https://github.com/Maison-de-la-Simulation/heraclespp.git
```

In case the repository has already been cloned by:

```bash
git clone https://gitlab.maisondelasimulation.fr/lrousselhard/nova.git
```

You can retrieve the dependencies (PDI, Kokkos,...) with:

```bash
git submodule init && git submodule update
```

## How to build with pre-installed dependencies

A straightforward way to build HERACLES++ is to assume that all dependencies are available in the environment. In this case, one can compile the Sod shock tube setup from the root of the project with the following commands:

```bash
cmake \
    -DBUILD_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DNovapp_SETUP=shock_tube \
    -DNovapp_NDIM=1 \
    -DNovapp_EOS=PerfectGas \
    -DNovapp_GRAVITY=Uniform \
    -DNovapp_GEOM=Cartesian \
    -DNovapp_inih_DEPENDENCY_POLICY=INSTALLED \
    -DNovapp_Kokkos_DEPENDENCY_POLICY=INSTALLED \
    -B build
cmake --build build --parallel 2
```

One can notice that HERACLES++ related options are being prefixed by `Novapp`.

The executable can be found in the directory `build/src/nova++`.

The execution needs an input file and is done with the following command:

```bash
./build/src/nova++ ./inputs/shock_tube.ini
```

For more complex compilation scenarios, please refer to the `toolchains` directory.

The results can be displayed with the Python script:

```bash
python3 ./test/view_shock_tube.py shock_tube_00000002.h5
```

## How to build with vendored dependencies

The library PDI is used for the I/O. The compilation needs to be done at the root of the directory. If PDI is already installed, this step is not necessary.

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
cmake --install build-pdi --prefix vendor/install_pdi
```

## How to cite

```bash
@misc{rousselhard2025heraclesmultidimensionaleuleriancode,
      title={HERACLES++: A multidimensional Eulerian code for exascale computing},
      author={Lou Roussel-Hard and Edouard Audit and Luc Dessart and Thomas Padioleau and Yushan Wang},
      year={2025},
      eprint={2503.04428},
      archivePrefix={arXiv},
      primaryClass={astro-ph.SR},
      url={https://arxiv.org/abs/2503.04428},
}
```
