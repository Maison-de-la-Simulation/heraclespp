# HERACLES++

## How to get sources

To get the full source of nova and external libraries:

```bash
git clone --recurse-submodules https://github.com/Maison-de-la-Simulation/heraclespp.git
```

In case the repository has already been cloned by

```bash
git clone https://gitlab.maisondelasimulation.fr/lrousselhard/nova.git
```

You can retrieve the dependencies (PDI, Kokkos,...) with

```bash
git submodule init && git submodule update
```

## How to build

A straightforward way to build Nova++ is to assume that all dependencies are available in the environment. In this case, one can compile the advection setup in a directory called `build` with the two following commands

```bash
cmake \
    -B build \
    -DBUILD_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DNovapp_SETUP=sedov1d \
    -DNovapp_NDIM=1 \
    -DNovapp_EOS=PerfectGas \
    -DNovapp_GRAVITY=Uniform \
    -DNovapp_GEOM=Cartesian \
    -DNovapp_Kokkos_DEPENDENCY_POLICY=INSTALLED
cmake --build build --parallel 2
```

One can notice that Nova++ related options are being prefixed by `Novapp`.

The executable can be found in the directory `build/src/nova++`.

For more complex compilation scenarios, please refer to the `toolchains` directory.


## Citation

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
