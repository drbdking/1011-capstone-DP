#!/bin/bash
#SBATCH --job-name=adv_fine_tune
#SBATCH --account=ds_ga_1011-2023fa
#SBATCH --partition=n1s8-v100-1
#SBATCH --open-mode=append
#SBATCH --output=/scratch/dc5116/1011-capstone-DP/output/slurm_output/%j_%x.out
#SBATCH --error=/scratch/dc5116/1011-capstone-DP/output/slurm_output/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue


singularity exec --bind /scratch --nv --overlay /scratch/dc5116/overlay-25GB-500K.ext3:rw /scratch/dc5116/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh
conda activate 2590-hw3
cd /scratch/dc5116/1011-capstone-DP
git pull
bash ./scripts/adv_fine_tune.sh
"