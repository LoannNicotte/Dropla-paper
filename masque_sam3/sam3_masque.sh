#!/bin/bash
# Lance sam3_masques.py sur AI-LAB (job Slurm, 1 GPU, conteneur Singularity).
#
# AVANT LA PREMIERE FOIS : il faut un conteneur avec Python 3.12 ou plus et PyTorch recent.
#   ls /ceph/container/pytorch/            # regarde s'il y a une image recente (Python 3.12+, PyTorch 2.7+)
# Sinon, telecharge-en une (10 a 20 minutes, une seule fois) :
#   mkdir -p ~/containers ~/.singularity/tmp ~/.singularity/cache
#   export SINGULARITY_TMPDIR=$HOME/.singularity/tmp SINGULARITY_CACHEDIR=$HOME/.singularity/cache
#   cd ~/containers && srun --mem=40G singularity pull docker://nvcr.io/nvidia/pytorch:25.06-py3
# puis mets le chemin du fichier .sif obtenu dans CONTAINER ci-dessous.
#
# UTILISATION (depuis le noeud de connexion, dans le dossier qui contient sam3_masques_ailab.py) :
#   sbatch lancer_sam3_masques.sh
#   squeue --me
#   tail -f sam3_masques_<numero>.out
# Test rapide : mets MAX_IMAGES = 20 dans sam3_masques_ailab.py. Reprise apres coupure : relance sbatch.

#SBATCH --job-name=sam3_masques
#SBATCH --output=sam3_masques.out
#SBATCH --error=sam3_masques.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

set -eo pipefail

# ----------------------------- REGLAGES -----------------------------
SCRIPT="sam3_masque.py"                              # script Python a lancer (dans le dossier d'ou tu fais sbatch)
CONTAINER="/ceph/container/pytorch/pytorch_26.09.sif"          # conteneur avec Python 3.12+ et PyTorch recent
PYLIBS="$HOME/pylibs_sam3"                                  # ou sont installes SAM 3 et ses paquets (une seule fois)
# --------------------------------------------------------------------

cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [ ! -f "$SCRIPT" ]; then
    echo "ERREUR : $SCRIPT introuvable dans $PWD (lance sbatch depuis le dossier du script)." >&2
    exit 1
fi
if [ ! -f "$CONTAINER" ]; then
    echo "ERREUR : conteneur introuvable : $CONTAINER (voir les explications en haut de ce fichier)." >&2
    exit 1
fi

mkdir -p "$PYLIBS"
export PYTHONUSERBASE="$PYLIBS"      # pip --user installe ici, et Python le retrouve aux lancements suivants
export PYTHONUNBUFFERED=1

echo "Noeud : $(hostname) ; demarrage : $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

singularity exec --nv "$CONTAINER" python3 "$SCRIPT"

echo "Fin : $(date)"