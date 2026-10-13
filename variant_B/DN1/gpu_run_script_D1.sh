#!/bin/bash
#SBATCH --account=p1156-25-1  # Project number
#SBATCH --job-name=gpu_test_run # Name of the job in SLURM
#SBATCH --partition=gpu
#SBATCH -o result.txt
#SBATCH -e error.txt
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=16 # Number of CPU tasks
##SBATCH --mem=60000 # Max memory allocated in MB
##SBATCH --nodelist=n142


module load singularity 


time singularity exec --nv -B /projects/p1156-25-1/ /projects/p1156-25-1/DimeNet/DimeNet.sif python3  DimeNet_in_vitro_D1.py









