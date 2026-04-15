#!/bin/bash
#SBATCH --job-name=scream_sweep_1gpu
#SBATCH --account=desi_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=%j_sweep_em_cathode_1gpu.out
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=akaXXXX@gmail.com

source /global/common/software/nersc/pe/conda/24.10.0/Miniforge3-24.7.1-0/etc/profile.d/conda.sh
conda activate /global/homes/p/XXXXa/.conda/envs/myenv

export TMPDIR=$PSCRATCH/tmp
mkdir -p $TMPDIR

cd /global/homes/p/XXXXa/SCREAM

# SWEEP_ID must be exported before sbatch, e.g.:
#   export SWEEP_ID="XXXXa/SCREAM_GD1_SWEEP/<id>"
#   sbatch slurm/sweep_em_cathode_gpu1.sh

python scripts/sweep_train.py \
    --sweep-id "$SWEEP_ID" \
    --count 50 \
    --seed-offset 0
