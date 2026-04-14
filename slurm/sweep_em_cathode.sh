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
#SBATCH --mail-user=akapratsos@gmail.com

source /global/common/software/nersc/pe/conda/24.10.0/Miniforge3-24.7.1-0/etc/profile.d/conda.sh
conda activate /global/homes/p/pratsosa/.conda/envs/myenv

export TMPDIR=$PSCRATCH/tmp
mkdir -p $TMPDIR

cd /global/homes/p/pratsosa/SCREAM

# SWEEP_ID must be exported before sbatch, e.g.:
#   export SWEEP_ID="pratsosa/SCREAM_GD1_SWEEP/<id>"
#   sbatch slurm/sweep_em_cathode.sh

for GPU_ID in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/sweep_train.py \
        --sweep-id "$SWEEP_ID" \
        --count 25 &
done

wait
