#!/bin/bash
#SBATCH --job-name=scpu
#SBATCH --output=%x.o%j
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=20              # thread number
#SBATCH --partition=cpu_short           # (see available partitions)

set -e

# To clean and load modules defined at the compile and link phases
module purge
module load gcc/11.2.0/gcc-4.8.5 hdf5/1.10.7/gcc-11.2.0-openmpi openmpi/4.1.1/gcc-11.2.0 cuda/11.7.0/gcc-11.2.0 cmake/3.21.4/gcc-11.2.0

# echo of commands
set -x

# To compute in the submission directory
cd "${SLURM_SUBMIT_DIR}"

# number of OpenMP threads
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Binding OpenMP threads on core
export OMP_PLACES=cores

# execution
./../bin/nova++ ../inputs/kelvin_helmholtz.ini
