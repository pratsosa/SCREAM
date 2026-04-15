#!/bin/bash
#SBATCH --job-name=scream_flow
#SBATCH --account=desi_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=%j_train_flow.out

source /global/common/software/nersc/pe/conda/24.10.0/Miniforge3-24.7.1-0/etc/profile.d/conda.sh
conda activate /global/homes/p/XXXXa/.conda/envs/myenv

cd /global/homes/p/XXXXa/SCREAM

srun python scripts/train_flow.py \
    --stream configs/streams/gd1.yaml \
    --epochs 200 \
    --batch-size 512 \
    --max-lr 3e-4 \
    --n-multiplier 4 \
    --seed 12345 \
    --patience 30