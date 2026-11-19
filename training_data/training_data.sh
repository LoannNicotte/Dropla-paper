#!/bin/bash
# Lance preparer_dataset_ailab.py sur AI-LAB (job Slurm, sans GPU, conteneur Singularity).
# Construit le dataset de fine-tuning (renumerotation des classes, decoupe train / val / test par video).
#
# UTILISATION (depuis le noeud de connexion, dans le dossier qui contient preparer_dataset_ailab.py) :
#   sbatch lancer_preparer_dataset.sh
#   cat preparer_dataset_<numero>.out
# Avant, ouvre preparer_dataset_ailab.py et verifie IMAGES_DIR, MASKS_DIR, OUTPUT_DIR et VIDEO_SPLITS.

#SBATCH --job-name=training_data
#SBATCH --output=training_data.out
#SBATCH --error=training_data.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00

set -eo pipefail

# ----------------------------- REGLAGES -----------------------------
SCRIPT="training_data.py"
CONTAINER="/ceph/container/pytorch/pytorch_26.09.sif"          # n'importe quel conteneur avec Python, numpy et pillow
# --------------------------------------------------------------------

cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [ ! -f "$SCRIPT" ]; then
    echo "ERREUR : $SCRIPT introuvable dans $PWD (lance sbatch depuis le dossier du script)." >&2
    exit 1
fi
if [ ! -f "$CONTAINER" ]; then
    echo "ERREUR : conteneur introuvable : $CONTAINER (voir lancer_sam3_masques.sh pour le telecharger)." >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
singularity exec "$CONTAINER" python3 "$SCRIPT"