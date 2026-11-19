#!/bin/bash
#SBATCH --job-name=comparaison
#SBATCH --output=comparaison.out
#SBATCH --error=comparaison.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

SCRIPT="comparaison_model.py"
CONTAINER="/ceph/container/pytorch/pytorch_26.09.sif"
export PYTHONUSERBASE="$HOME/pylibs_train"     # la ou sont opencv et transformers
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR:-$PWD}"
singularity exec --nv "$CONTAINER" python3 "$SCRIPT"