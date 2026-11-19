"""Cree un dataset SegFormer : val fixe annotee a la main, train = annotations manuelles + SAM 3.

Entrees (meme nom partout, seule l'extension change : img12.jpg <-> img12.png) :
  - IMAGES_DIR     : toutes les images (.jpg), val comprise
  - VAL_MASKS_DIR  : annotations a la main des images de VALIDATION (.png, noir = fond, rouge = tronc)
  - MANUAL_DIR     : annotations a la main pour le train (.png, noir = fond, rouge = tronc)
  - SAM3_DIR       : masques SAM 3 (.png, 0 = fond, 1 = tronc)

Principe :
  - val   : une image par masque de VAL_MASKS_DIR (l'image est prise dans IMAGES_DIR).
  - train : TRAIN_SIZE images, jamais une image de val (ni en manuel, ni en SAM 3).
            POURCENTAGE % ont un masque manuel, le reste est complete avec des masques SAM 3.
  Les images manuelles sont tirees dans un ordre fixe : l'essai a 10 % contient celles de l'essai a 5 %.

Sortie (OUTPUT_DIR, efface puis recree a chaque lancement) :
  images/{train,val}/imgX.jpg, masks/{train,val}/imgX.png (0 = fond, 1 = tronc), liste.csv
"""

import csv
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

# ----------------------------- REGLAGES -----------------------------
POURCENTAGE = 10                     # % des images de train avec un masque manuel (le reste = SAM 3)
TRAIN_SIZE = 2750                   # nombre total d'images de train

IMAGES_DIR = "~/Dropla/paper/data/images"
VAL_MASKS_DIR = "~/Dropla/paper/data/images_val"      # masques manuels de la val
TRAIN_MANUAL_DIR = "~/Dropla/paper/data/images_train"      # masques manuels pour le train
SAM3_DIR = "~/Dropla/paper/data/images_sam3"
OUTPUT_DIR = f"~/Dropla/paper/data/dataset_manuel_{POURCENTAGE}"
SEED = 0                            # garde le meme pour tous les essais
# --------------------------------------------------------------------

images_dir, val_masks_dir, manual_dir, sam3_dir, out = (
    Path(p).expanduser() for p in (IMAGES_DIR, VAL_MASKS_DIR, TRAIN_MANUAL_DIR, SAM3_DIR, OUTPUT_DIR))


def manual_to_mask(path):
    """Annotation a la main -> masque 0/1 (tout pixel rouge = tronc)."""
    rgb = np.array(Image.open(path).convert("RGB"))
    return (rgb[:, :, 0] > 0).astype(np.uint8)


def save(image_path, mask, split):
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "masks" / split).mkdir(parents=True, exist_ok=True)
    shutil.copy(image_path, out / "images" / split / image_path.name)
    Image.fromarray(mask).save(out / "masks" / split / (image_path.stem + ".png"))


# ---- Val : une image par masque de VAL_MASKS_DIR ----
val_stems = {p.stem for p in val_masks_dir.glob("*.png")}
if not val_stems:
    raise FileNotFoundError(f"Aucun masque .png dans {val_masks_dir}")
no_image = sorted(s for s in val_stems if not (images_dir / (s + ".jpg")).exists())
if no_image:
    raise FileNotFoundError(f"{len(no_image)} masques de val sans image dans {images_dir}, ex. {no_image[:3]}")
val_images = sorted(images_dir / (s + ".jpg") for s in val_stems)

# ---- Train : tout sauf la val ----
train_images = {p.stem: p for p in images_dir.glob("*.jpg") if p.stem not in val_stems}
manual_stems = sorted(p.stem for p in manual_dir.glob("*.png") if p.stem in train_images)
random.Random(SEED).shuffle(manual_stems)                       # ordre fixe : essais emboites

n_manual = round(POURCENTAGE / 100 * TRAIN_SIZE)
if n_manual > len(manual_stems):
    raise ValueError(f"{POURCENTAGE} % de {TRAIN_SIZE} = {n_manual} images manuelles, "
                     f"mais seulement {len(manual_stems)} disponibles hors val.")
chosen_manual = manual_stems[:n_manual]

sam3_stems = sorted(s for s in train_images if s not in set(chosen_manual) and (sam3_dir / (s + ".png")).exists())
random.Random(SEED).shuffle(sam3_stems)
n_sam3 = TRAIN_SIZE - n_manual
if n_sam3 > len(sam3_stems):
    raise ValueError(f"Il faut {n_sam3} masques SAM 3 pour completer, mais seulement {len(sam3_stems)} disponibles hors val.")
chosen_sam3 = sam3_stems[:n_sam3]

print(f"{len(val_images)} images de val | {len(train_images)} images hors val, "
      f"dont {len(manual_stems)} annotees a la main", flush=True)

# ---- Ecriture ----
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)

rows = []
total = len(val_images) + len(chosen_manual) + len(chosen_sam3)
PRINT_EVERY = 100                   # affiche la progression toutes les N images


def progress(step):
    if len(rows) % PRINT_EVERY == 0 or len(rows) == total:
        print(f"  {len(rows)}/{total} images ({step})", flush=True)


print("Copie de la val ...", flush=True)
for path in val_images:
    save(path, manual_to_mask(val_masks_dir / (path.stem + ".png")), "val")
    rows.append([path.name, "val", "manuel"])
    progress("val")

print("Copie du train (manuel) ...", flush=True)
for stem in chosen_manual:
    save(train_images[stem], manual_to_mask(manual_dir / (stem + ".png")), "train")
    rows.append([train_images[stem].name, "train", "manuel"])
    progress("train manuel")

print("Copie du train (SAM 3) ...", flush=True)
for stem in chosen_sam3:
    save(train_images[stem], np.array(Image.open(sam3_dir / (stem + ".png"))), "train")
    rows.append([train_images[stem].name, "train", "sam3"])
    progress("train SAM 3")

with open(out / "liste.csv", "w", newline="", encoding="utf-8") as file:
    csv.writer(file).writerows([["image", "split", "source"]] + rows)

print(f"train : {len(chosen_manual) + len(chosen_sam3)} images "
      f"({len(chosen_manual)} manuelles = {POURCENTAGE} %, {len(chosen_sam3)} SAM 3)")
print(f"val   : {len(val_images)} images manuelles")
print(f"Dataset cree dans {out}")