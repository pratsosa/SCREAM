#!/bin/bash
#SBATCH --job-name=scream_em_cathode
#SBATCH --account=desi_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=%j_train_em_cathode.out

source /global/common/software/nersc/pe/conda/24.10.0/Miniforge3-24.7.1-0/etc/profile.d/conda.sh
conda activate /global/homes/p/pratsosa/.conda/envs/myenv

cd /global/homes/p/pratsosa/SCREAM

srun python scripts/train_em_cathode.py \
    --stream configs/streams/gd1.yaml \
    --experiment configs/experiments/em_cathode_mlp.yaml
