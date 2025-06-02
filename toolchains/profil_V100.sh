#!/bin/bash

## PDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
##
## SPDX-License-Identifier: MIT

#SBATCH --job-name=sgpu
#SBATCH --output=%x.o%j
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_test # V100 : gpu/gpu_test || A100 : gpua100 || P100 : gpup100

## GPU allocation
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

set -e

## To clean and load modules defined at the compile and link phases
module purge
module load gcc/11.2.0/gcc-4.8.5 \
	hdf5/1.10.7/gcc-11.2.0-openmpi \
	openmpi/4.1.1/gcc-11.2.0 \
	cuda/12.2.1/gcc-11.2.0 \
	cmake/3.28.3/gcc-11.2.0


cd ..
. vendor/install_pdi/share/pdi/env.zsh

export KOKKOS_TOOLS_LIBS=/gpfs/users/roaldese/session0/kokkos-tools/build/profiling/nvtx-connector/libkp_nvtx_connector.so

# echo of commands
set -x

## To compute in the submission directory
cd "${SLURM_SUBMIT_DIR}"

## execution
srun ./../build/src/nova++ ../inputs/rayleigh_taylor3d.ini

## mpirun si plusieurs MPI

# nsys start ?
nsys profile --trace=cuda,nvtx ./../build/src/nova++ ../inputs/rayleigh_taylor3d.ini

mkdir -p /gpfs/users/roaldese/Heraclespp/heraclespp_ers/outputs/

rm *.h5 *.xmf
