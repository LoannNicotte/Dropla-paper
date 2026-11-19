#!/bin/bash
#SBATCH --job-name=finetune_b0_0
#SBATCH --output=b0_0.out
#SBATCH --error=b0_0.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00

SCRIPT="finetune_b0.py"
CONTAINER="/ceph/container/pytorch/pytorch_26.09.sif"
export PYTHONUSERBASE="$HOME/pylibs_train"
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR:-$PWD}"
singularity exec --nv "$CONTAINER" python3 "$SCRIPT"