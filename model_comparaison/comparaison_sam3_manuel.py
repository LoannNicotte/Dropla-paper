"""IoU des troncs entre les masques SAM 3 et les annotations manuelles (val).

Compare les images presentes dans les DEUX dossiers (meme nom).
Un pixel est un tronc si sa valeur est > 0 (marche pour 0/1 et pour le rouge sur noir).

Affiche :
  - IoU generale : tous les pixels de toutes les images ensemble (comme pendant l'entrainement)
  - IoU moyenne par image
  - precision et rappel de SAM 3
et enregistre le detail par image dans CSV_PATH.
"""

import csv
from pathlib import Path

import numpy as np
from PIL import Image

# ----------------------------- REGLAGES -----------------------------
MANUAL_DIR = "~/Dropla/paper/data/images_val"          # annotations a la main (verite terrain)
SAM3_DIR = "~/Dropla/paper/data/images_sam3_val"       # masques SAM 3
CSV_PATH = "~/Dropla/paper/model_comparaison/iou_sam3_val.csv"     # detail par image
# --------------------------------------------------------------------


def load_trunk(path, size=None):
    """Masque booleen des troncs (si couleur : canal rouge > 0, sinon valeur > 0)."""
    image = Image.open(path)
    if size is not None and image.size != size:
        image = image.resize(size, Image.NEAREST)
    array = np.array(image)
    if array.ndim == 3:
        array = array[:, :, 0]
    return array > 0


manual_dir, sam3_dir = Path(MANUAL_DIR).expanduser(), Path(SAM3_DIR).expanduser()
manual = {p.stem: p for p in manual_dir.glob("*.png")}
sam3 = {p.stem: p for p in sam3_dir.glob("*.png")}
common = sorted(manual.keys() & sam3.keys())
print(f"{len(manual)} masques manuels, {len(sam3)} masques SAM 3, {len(common)} en commun")
if not common:
    raise FileNotFoundError("Aucune image en commun entre les deux dossiers (verifie les noms et les chemins).")

total_inter = total_union = total_sam3 = total_manual = 0
rows = []
for name in common:
    truth = load_trunk(manual[name])
    pred = load_trunk(sam3[name], size=(truth.shape[1], truth.shape[0]))   # meme taille que le manuel
    inter = int((truth & pred).sum())
    union = int((truth | pred).sum())
    total_inter += inter
    total_union += union
    total_sam3 += int(pred.sum())
    total_manual += int(truth.sum())
    rows.append([name, round(inter / union, 4) if union else 1.0, round(100 * truth.mean(), 2)])

iou_global = total_inter / max(1, total_union)
iou_mean = float(np.mean([r[1] for r in rows]))
precision = total_inter / max(1, total_sam3)
recall = total_inter / max(1, total_manual)

print(f"\nIoU tronc generale     : {iou_global:.4f}")
print(f"IoU tronc moyenne      : {iou_mean:.4f}  (moyenne des {len(rows)} images)")
print(f"Precision SAM 3        : {precision:.4f}  (part des troncs SAM 3 qui sont vraiment des troncs)")
print(f"Rappel SAM 3           : {recall:.4f}  (part des vrais troncs trouves par SAM 3)")

rows.sort(key=lambda r: r[1])
print("\n5 images les moins bonnes :")
for name, iou, percent in rows[:5]:
    print(f"  {name:<20} IoU {iou:.4f}  ({percent:.1f} % de tronc dans le masque manuel)")

csv_path = Path(CSV_PATH).expanduser()
with open(csv_path, "w", newline="", encoding="utf-8") as file:
    csv.writer(file).writerows([["image", "iou_tronc", "pourcent_tronc_manuel"]] + rows)
print(f"\nDetail par image : {csv_path}")