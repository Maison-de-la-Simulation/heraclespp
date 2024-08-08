#!/bin/bash
#SBATCH --account=c1615137
#SBATCH --job-name=nova-run
#SBATCH --output=%x.o%j
#SBATCH --nodes=8
#SBATCH --exclusive
#SBATCH --constraint=MI250
#SBATCH --time=23:00:00

# load modules here
module load cpe/23.12 craype-x86-trento craype-accel-amd-gfx90a PrgEnv-amd cray-hdf5-parallel

# To compute in the submission directory
cd ${SLURM_SUBMIT_DIR}

# execution
srun --ntasks=64 --gpus-per-node=8 ./../bin/nova++ ../inputs/v1d_3d_5e2_768p_1_256.ini
