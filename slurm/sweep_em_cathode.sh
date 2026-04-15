#!/bin/bash
#SBATCH --job-name=scream_sweep
#SBATCH --account=desi_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=%j_sweep_em_cathode.out
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=akaXXXX@gmail.com

source /global/common/software/nersc/pe/conda/24.10.0/Miniforge3-24.7.1-0/etc/profile.d/conda.sh
conda activate /global/homes/p/XXXXa/.conda/envs/myenv

export TMPDIR=$PSCRATCH/tmp
mkdir -p $TMPDIR

cd /global/homes/p/XXXXa/SCREAM

# SWEEP_ID must be exported before sbatch, e.g.:
#   export SWEEP_ID="XXXXa/SCREAM_GD1_SWEEP/<id>"
#   sbatch slurm/sweep_em_cathode.sh
#
# Seed assignment for 50-run sweeps: agents cover non-overlapping seed ranges.
#   GPU 0: seeds  1-13 (offset=0,  count=13)
#   GPU 1: seeds 14-26 (offset=13, count=13)
#   GPU 2: seeds 27-38 (offset=26, count=12)
#   GPU 3: seeds 39-50 (offset=38, count=12)
# Adjust offsets/counts proportionally for sweeps with a different run_cap.

CUDA_VISIBLE_DEVICES=0 python scripts/sweep_train.py \
    --sweep-id "$SWEEP_ID" --count 13 --seed-offset 0  &

CUDA_VISIBLE_DEVICES=1 python scripts/sweep_train.py \
    --sweep-id "$SWEEP_ID" --count 13 --seed-offset 13 &

CUDA_VISIBLE_DEVICES=2 python scripts/sweep_train.py \
    --sweep-id "$SWEEP_ID" --count 12 --seed-offset 26 &

CUDA_VISIBLE_DEVICES=3 python scripts/sweep_train.py \
    --sweep-id "$SWEEP_ID" --count 12 --seed-offset 38 &

wait
