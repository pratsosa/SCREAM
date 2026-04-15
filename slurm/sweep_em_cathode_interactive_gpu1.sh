#!/bin/bash
# Interactive single-GPU sweep launcher — use after acquiring a node with salloc.
#
# Example salloc command:
#   salloc --account=desi_g --constraint=gpu --qos=interactive --nodes=1 --gpus=1 --cpus-per-task=16 --time=04:00:00
#
# Then run:
#   export SWEEP_ID="pratsosa/SCREAM_GD1_SWEEP/<id>"
#   bash slurm/sweep_em_cathode_interactive_gpu1.sh

source /global/common/software/nersc/pe/conda/24.10.0/Miniforge3-24.7.1-0/etc/profile.d/conda.sh
conda activate /global/homes/p/pratsosa/.conda/envs/myenv

export TMPDIR=$PSCRATCH/tmp
mkdir -p $TMPDIR

cd /global/homes/p/pratsosa/SCREAM

python scripts/sweep_train.py \
    --sweep-id "$SWEEP_ID" \
    --count 50 \
    --seed-offset 0
