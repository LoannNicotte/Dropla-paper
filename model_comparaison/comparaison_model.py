"""Compare N modeles SegFormer (troncs) sur les images de validation.

Ne traite que les images dont le masque manuel existe dans MANUAL_DIR.
Pour chaque image, cree une planche :
  image | masque manuel | masque SAM 3 | modele 1 | modele 2 | ...
Les troncs sont en rouge par-dessus l'image (le reste n'est pas modifie).
Chaque vignette porte son nom en haut, et en bas son IoU tronc par rapport au masque manuel.
A la fin, affiche l'IoU generale de SAM 3 et de chaque modele sur toutes les images traitees.
"""

import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

# ----------------------------- REGLAGES -----------------------------
IMAGES_DIR = "~/Dropla/paper/data/images"              # images (.jpg)
MANUAL_DIR = "~/Dropla/paper/data/images_val"          # masques manuels de la val (rouge = tronc)
SAM3_DIR = "~/Dropla/paper/data/images_sam3_val"       # masques SAM 3 (0/1)
OUTPUT_DIR = "~/Dropla/paper/comparaison/comp1"

# nom affiche sur la vignette -> dossier du modele (le dossier "best" cree par l'entrainement)
MODELS = {
    "0 % manuel": "~/Dropla/paper/finetune/finetune_b0_0/best",
    "0 % manuel scratch": "~/Dropla/paper/finetune/finetune_b0_0_SCRATCH/best",
    "10 % manuel scratch": "~/Dropla/paper/finetune/finetune_b0_10_SCRATCH/best",
}
IMAGE_SIZE = (1280, 720)            # (largeur, hauteur) donnee au modele : la meme qu'a l'entrainement
COLUMNS = None                      # vignettes par ligne ; None = automatique (grille la plus carree possible)
ALPHA = 0.5                         # opacite du rouge (0 = invisible, 1 = rouge plein)
MAX_IMAGES = 20                   # ex: 20 pour un test rapide (tirees au hasard) ; None = toutes
SEED = 1                            # meme graine = memes images a chaque lancement
# --------------------------------------------------------------------

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RED = np.array([0, 0, 255], dtype=np.float32)          # BGR


def load_mask(path, shape):
    """Masque booleen des troncs a la taille de l'image (couleur : canal rouge > 0, sinon valeur > 0)."""
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask.ndim == 3:
        mask = mask[:, :, 2]                            # canal rouge (OpenCV lit en BGR)
    mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def predict(model, bgr):
    """Masque des troncs predit (booleens) a la taille de l'image d'origine."""
    rgb = cv2.cvtColor(cv2.resize(bgr, IMAGE_SIZE), cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255 - MEAN) / STD
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(pixel_values=x).logits
        logits = F.interpolate(logits, size=bgr.shape[:2], mode="bilinear")
    return (logits.argmax(1)[0] == 1).cpu().numpy()


def overlay(bgr, trunk):
    out = bgr.copy()
    out[trunk] = ((1 - ALPHA) * bgr[trunk] + ALPHA * RED).astype(np.uint8)
    return out


def iou(pred, truth):
    union = (pred | truth).sum()
    return (pred & truth).sum() / union if union else 1.0


def label(img, text, bottom=False):
    scale = img.shape[0] / 600
    thickness = max(1, round(2 * scale))
    position = (round(15 * scale), img.shape[0] - round(20 * scale) if bottom else round(45 * scale))
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img


def grid(tiles):
    columns = COLUMNS or math.ceil(math.sqrt(len(tiles)))
    rows = math.ceil(len(tiles) / columns)
    tiles = tiles + [np.zeros_like(tiles[0])] * (rows * columns - len(tiles))
    return np.vstack([np.hstack(tiles[r * columns:(r + 1) * columns]) for r in range(rows)])


images_dir, manual_dir, sam3_dir, out_dir = (Path(p).expanduser() for p in (IMAGES_DIR, MANUAL_DIR, SAM3_DIR, OUTPUT_DIR))
out_dir.mkdir(parents=True, exist_ok=True)

# Images = celles qui ont un masque manuel (et dont l'image existe)
images = []
for mask_path in sorted(manual_dir.glob("*.png")):
    candidates = [images_dir / (mask_path.stem + ext) for ext in (".jpg", ".jpeg", ".png")]
    image_path = next((p for p in candidates if p.exists()), None)
    if image_path is None:
        print(f"  image introuvable pour {mask_path.name}, ignoree")
    else:
        images.append(image_path)
if MAX_IMAGES is not None and MAX_IMAGES < len(images):
    images = sorted(random.Random(SEED).sample(images, MAX_IMAGES))
print(f"{len(images)} images avec masque manuel, {len(MODELS)} modeles, appareil : {DEVICE}", flush=True)

models = {}
for name, path in MODELS.items():
    path = Path(path).expanduser()
    if not (path / "config.json").exists():
        raise FileNotFoundError(f"Modele '{name}' introuvable : {path} (verifie le chemin, majuscules comprises)")
    models[name] = SegformerForSemanticSegmentation.from_pretrained(path).to(DEVICE).eval()
    print(f"Modele charge : {name}", flush=True)

totals = {name: [0, 0] for name in ["SAM 3", *models]}     # [intersection, union] sur toutes les images


def tile(bgr, trunk, truth, name):
    """Vignette : troncs en rouge, nom en haut, IoU par rapport au manuel en bas."""
    totals[name][0] += int((trunk & truth).sum())
    totals[name][1] += int((trunk | truth).sum())
    img = label(overlay(bgr, trunk), name)
    return label(img, f"IoU {iou(trunk, truth):.3f}", bottom=True)


for i, path in enumerate(images, 1):
    bgr = cv2.imread(str(path))
    truth = load_mask(manual_dir / (path.stem + ".png"), bgr.shape)
    tiles = [label(bgr.copy(), "Image")]
    tiles.append(label(overlay(bgr, truth), "Manuel (reference)"))
    sam3_path = sam3_dir / (path.stem + ".png")
    if sam3_path.exists():
        tiles.append(tile(bgr, load_mask(sam3_path, bgr.shape), truth, "SAM 3"))
    else:
        tiles.append(label(bgr.copy(), "SAM 3 : absent"))
    for name, model in models.items():
        tiles.append(tile(bgr, predict(model, bgr), truth, name))
    cv2.imwrite(str(out_dir / (path.stem + ".jpg")), grid(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if i % 20 == 0 or i == len(images):
        print(f"  {i}/{len(images)}", flush=True)

print("\nIoU tronc generale par rapport au masque manuel :")
for name, (inter, union) in totals.items():
    print(f"  {name:<25} {inter / max(1, union):.4f}")
print(f"Planches enregistrees dans {out_dir}")