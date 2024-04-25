#!/bin/bash
#SBATCH --account=cin4698
#SBATCH --job-name=nova-run
#SBATCH --output=%x.o%j
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --constraint=MI250
#SBATCH --time=10:00:00

# load modules here
module load cpe/23.05 craype-x86-trento craype-accel-amd-gfx90a PrgEnv-gnu amd-mixed/5.5.1 rocm/5.2.3 cray-hdf5-parallel/1.12.2.1 cray-mpich/8.1.24

# To compute in the submission directory
cd ${SLURM_SUBMIT_DIR}

# execution
srun --ntasks=8 --gpus-per-node=8 ./../bin/nova++ ../inputs/v1d_3d_5e2_768p.ini  
