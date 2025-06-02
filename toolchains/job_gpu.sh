#!/bin/bash

# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

#SBATCH --job-name=sgpu
#SBATCH --output=%x.o%j
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_test           # (see available partitions)
#SBATCH --gres=gpu:1

set -e

# To clean and load modules defined at the compile and link phases
module purge
module load gcc/11.2.0/gcc-4.8.5 hdf5/1.10.7/gcc-11.2.0-openmpi openmpi/4.1.1/gcc-11.2.0 cuda/12.2.1/gcc-11.2.0 cmake/3.28.3/gcc-11.2.0

# echo of commands
set -x

# To compute in the submission directory
cd "${SLURM_SUBMIT_DIR}"

# execution
./../build/src/nova++ ../inputs/sedov2d.ini

mv *.h5 *.xmf sgpu* /gpfs/users/roaldese/Heraclespp/heraclespp_ers/outputs/
