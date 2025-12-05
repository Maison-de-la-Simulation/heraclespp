#!/bin/bash

# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

#SBATCH --account=cin4698
#SBATCH --job-name=HERACLES++-run
#SBATCH --output=%x.o%j
#SBATCH --nodes=16
#SBATCH --exclusive
#SBATCH --constraint=MI250
#SBATCH --time=05:59:00

set -e

# load modules here
module load cpe/23.05 craype-x86-trento craype-accel-amd-gfx90a PrgEnv-gnu amd-mixed/5.5.1 rocm/5.2.3 cray-hdf5-parallel/1.12.2.1 cray-mpich/8.1.24

set -x

# To compute in the submission directory
cd "${SLURM_SUBMIT_DIR}"

# execution
srun -N16 -n128 -c2 --account=cin4698 --constraint=MI250 --gpus-per-node=8 --time=05:59:00 ./../bin/heracles++ ../inputs/rayleigh_taylor3d_sph.ini
