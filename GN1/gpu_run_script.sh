#!/bin/bash
#SBATCH --account=p1156-25-1  # Project number
#SBATCH --job-name=GemNet # Name of the job in SLURM
#SBATCH --partition=gpu
#SBATCH -o result.txt
#SBATCH -e error.txt
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=128G
##SBATCH --nodelist=n142


module load singularity 

time singularity exec --nv -B /projects/p1156-25-1/ /projects/p1156-25-1/DimeNet/GemNet.sif python3 GemNet_in_vitro_D1.py



