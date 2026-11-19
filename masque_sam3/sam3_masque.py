"""SAM 3 : masque des troncs d'arbres pour toutes les images d'un dossier.
Masque PNG : 0 = fond, 1 = tronc.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ----------------------------- REGLAGES -----------------------------
INPUT_DIR = "/ceph/home/student.aau.dk/ek03mz/Dropla/paper/data/images/"
CHECKPOINT_PATH = "~/Dropla/sam3.pt"
OUTPUT_DIR = "~/Dropla/paper/data/images_sam3"
PROMPT = "tree trunk"
CONFIDENCE_THRESHOLD = 0.35
TRUNK_VALUE = 1
# --------------------------------------------------------------------

input_dir = Path(INPUT_DIR).expanduser()
output_dir = Path(OUTPUT_DIR).expanduser()
output_dir.mkdir(parents=True, exist_ok=True)

model = build_sam3_image_model(device="cuda", checkpoint_path=str(Path(CHECKPOINT_PATH).expanduser()))
processor = Sam3Processor(model, device="cuda", confidence_threshold=CONFIDENCE_THRESHOLD)

images = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
print(f"{len(images)} images")

for i, path in enumerate(images, 1):
    out_path = output_dir / ("__".join(path.relative_to(input_dir).with_suffix(".png").parts))
    if out_path.exists():
        continue

    image = Image.open(path).convert("RGB")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        state = processor.set_text_prompt(prompt=PROMPT, state=state)
        masks = state["masks"]                      # (N, 1, H, W)

    mask = np.zeros((image.height, image.width), dtype=np.uint8)
    if len(masks) > 0:
        mask[masks[:, 0].any(dim=0).cpu().numpy()] = TRUNK_VALUE
    Image.fromarray(mask).save(out_path)

    if i % 50 == 0:
        print(f"{i}/{len(images)}", flush=True)

print(f"Termine. Masques dans {output_dir}")