#!/bin/bash
#SBATCH --job-name=iou_sam3
#SBATCH --output=iou_sam3.out
#SBATCH --error=iou_sam3.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:30:00

SCRIPT="comparaison_sam3_manuel.py"
CONTAINER="/ceph/container/pytorch/pytorch_26.09.sif"
export PYTHONUSERBASE="$HOME/pylibs_train"
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR:-$PWD}"
singularity exec "$CONTAINER" python3 "$SCRIPT"