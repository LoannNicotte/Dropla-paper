"""Fine-tuning de SegFormer B0 sur les troncs d'arbres (0 = fond, 1 = tronc, 255 = ignore), sur tous les GPU visibles.

Lancer avec : python3 finetune_troncs_4gpu.py   (et #SBATCH --gres=gpu:4 dans le .sh)

Dataset : images/{train,val}/<nom>.jpg et masks/{train,val}/<nom>.png
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerConfig, SegformerForSemanticSegmentation

# ----------------------------- REGLAGES -----------------------------
pourcentage_data = 10
FROM_SCRATCH = True                # True = poids aleatoires (prend l'architecture de PRETRAINED_MODEL, pas ses poids)
DATASET_DIR = "~/Dropla/paper/data/dataset_manuel_" + str(pourcentage_data)
OUTPUT_DIR = "~/Dropla/paper/finetune/finetune_b0_" + str(pourcentage_data)
if FROM_SCRATCH:
	OUTPUT_DIR = OUTPUT_DIR + "_SCRATCH"
PRETRAINED_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
IMAGE_SIZE = (1280, 720)             # (largeur, hauteur), multiples de 32
EPOCHS = 50
BATCH_SIZE = 8                     # lot TOTAL, partage entre les GPU (32 / 4 = 8 par GPU)
LEARNING_RATE = 6e-5
CLASS_WEIGHTS = [1.0, 5.0]          # poids [fond, tronc] : plus le tronc est lourd, plus ses erreurs comptent
# --------------------------------------------------------------------

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEVICE = "cuda"

class TrunkDataset(Dataset):
    def __init__(self, split, train):
        root = Path(DATASET_DIR).expanduser()
        self.images = sorted(p for p in (root / "images" / split).iterdir()
                             if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        self.masks_dir = root / "masks" / split
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = cv2.cvtColor(cv2.imread(str(self.images[i])), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.masks_dir / (self.images[i].stem + ".png")), cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, IMAGE_SIZE)
        mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
        if self.train and random.random() < 0.5:        # miroir horizontal
            image, mask = image[:, ::-1], mask[:, ::-1]
        image = (image.astype(np.float32) / 255 - MEAN) / STD
        return torch.from_numpy(image.transpose(2, 0, 1).copy()), torch.from_numpy(mask.astype(np.int64))


def trunk_iou(model, loader):
    model.eval()
    inter = union = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            logits = model(pixel_values=images, return_dict=False)[0]
            pred = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear").argmax(1)
            valid = masks != 255
            inter += ((pred == 1) & (masks == 1) & valid).sum().item()
            union += (((pred == 1) | (masks == 1)) & valid).sum().item()
    return inter / max(1, union)


train_loader = DataLoader(TrunkDataset("train", True), batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
val_loader = DataLoader(TrunkDataset("val", False), batch_size=BATCH_SIZE, num_workers=4)

LABELS = {"id2label": {0: "fond", 1: "tronc"}, "label2id": {"fond": 0, "tronc": 1}}
if FROM_SCRATCH:
    # meme architecture que PRETRAINED_MODEL, mais poids aleatoires
    config = SegformerConfig.from_pretrained(PRETRAINED_MODEL, num_labels=2, **LABELS)
    model = SegformerForSemanticSegmentation(config).to(DEVICE)
    print("Entrainement from scratch (poids aleatoires)", flush=True)
else:
    model = SegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED_MODEL, num_labels=2, ignore_mismatched_sizes=True, **LABELS).to(DEVICE)
    print(f"Fine-tuning depuis {PRETRAINED_MODEL}", flush=True)
print(f"{torch.cuda.device_count()} GPU utilises", flush=True)
model = torch.nn.DataParallel(model)
weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

best = 0.0
for epoch in range(EPOCHS):
    model.train()
    total = 0.0
    for images, masks in train_loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        logits = model(pixel_values=images).logits
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear")
        loss = F.cross_entropy(logits, masks, weight=weights, ignore_index=255)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()

    iou = trunk_iou(model, val_loader)
    print(f"Epoque {epoch + 1}/{EPOCHS} : perte {total / len(train_loader):.4f}, IoU tronc val {iou:.4f}", flush=True)
    if iou > best:
        best = iou
        model.module.save_pretrained(Path(OUTPUT_DIR).expanduser() / "best")

print(f"Termine. Meilleure IoU tronc : {best:.4f}")