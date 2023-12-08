#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=0-20:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name adv_ft
#SBATCH --output /gpfs/scratch/changd14/1011-capstone-DP/output/slurm_output/adv_ft-%J.log 

module load singularity/3.1
singularity exec --bind /scratch --nv --overlay /gpfs/scratch/changd14/overlay-25GB-500K.ext3:rw /gpfs/scratch/changd14/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
conda activate 2590-hw3
cd /gpfs/scratch/changd14/1011-capstone-DP
git pull
bash ./scripts/adv_fine_tune.sh
"