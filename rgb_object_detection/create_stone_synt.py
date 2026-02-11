import json
import random
import os
import numpy as np
from pycocotools import mask as maskUtils
from PIL import Image, ImageEnhance, ImageFilter

# Paths
COCO_JSON = "test/_annotations.coco.json"
IMAGE_DIR = "test/"
BACKGROUND_DIR = "container_backgrounds/"
OUTPUT_IMG_DIR = "output/images/"
OUTPUT_LABEL_DIR = "output/labels/"
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# Function to load COCO annotations
with open(COCO_JSON, "r") as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}
cats = {c["id"]: c["name"] for c in coco["categories"]}

anns_by_image = {}
for ann in coco["annotations"]:
    anns_by_image.setdefault(ann["image_id"], []).append(ann)

# Function to convert COCO to mask
def ann_to_mask(ann, h, w):
    seg = ann["segmentation"]
    if isinstance(seg, list):
        rles = maskUtils.frPyObjects(seg, h, w)
        rle = maskUtils.merge(rles)
    elif isinstance(seg["counts"], list):
        rle = maskUtils.frPyObjects(seg, h, w)
    else:
        rle = seg
    return maskUtils.decode(rle).astype(np.uint8)

# Function to apply augmentations before pasting
def random_transform(obj):
    # random scale
    scale = random.uniform(0.05, 0.25)
    new_w = max(20, int(obj.width * scale))
    new_h = max(20, int(obj.height * scale))
    new_w = min(80, new_w)
    new_h = min(80, new_h)

    obj = obj.resize((new_w, new_h), Image.LANCZOS)

    # random rotation (+/- 25 deg)
    angle = random.uniform(-25, 25)
    obj = obj.rotate(angle, expand=True)

    # alpha blending (simulated shadow / brightness variation)
    alpha_factor = random.uniform(0.9, 1.0)
    alpha = obj.split()[-1].point(lambda px: px * alpha_factor)
    obj.putalpha(alpha)

    # blur for realism
    obj = obj.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    return obj

# Loop through images
counter = 0
max_images = np.inf

for image_id, anns in anns_by_image.items():
    if counter >= max_images:
        break
    img_meta = images[image_id]
    file_name = img_meta["file_name"]
    if not file_name.lower().endswith((".jpg", ".jpeg")):
        continue
    
    img_path = os.path.join(IMAGE_DIR, file_name)
    src_img = Image.open(img_path).convert("RGBA")

    # Choose background
    valid_ext = (".jpg", ".jpeg", ".png")
    bg_candidates = [f for f in os.listdir(BACKGROUND_DIR) if f.lower().endswith(valid_ext)]
    
    bg_file = random.choice(bg_candidates)
    bg = Image.open(os.path.join(BACKGROUND_DIR, bg_file)).convert("RGBA")
    bg = bg.resize((src_img.width, src_img.height))

    # YOLO annotations for this output image
    yolo_lines = []

    for ann in anns:
        cat_id = ann["category_id"]

        # COCO mask
        mask = ann_to_mask(ann, img_meta["height"], img_meta["width"])
        mask_img = Image.fromarray(mask * 255).convert("L")

        # Object RGBA
        obj = src_img.copy()
        obj.putalpha(mask_img)

        # Crop to bbox
        x, y, w, h = [int(i) for i in ann["bbox"]]
        obj = obj.crop((x, y, x+w, y+h))

        if obj.getbbox() is None:
            continue

        # Transform (scale, rotate, alpha, blur)
        obj = random_transform(obj)

        # Random placement
        max_x = bg.width - obj.width
        max_y = bg.height - obj.height
        if max_x <= 0 or max_y <= 0:
            continue

        px = random.randint(0, max_x)
        py = random.randint(0, max_y)

        # Paste
        bg.paste(obj, (px, py), obj)

        # YOLO bounding box from placement
        x_center = (px + obj.width / 2) / bg.width
        y_center = (py + obj.height / 2) / bg.height
        w_norm = obj.width / bg.width
        h_norm = obj.height / bg.height

        line = f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
        yolo_lines.append(line)

    # Save only if something was pasted
    if yolo_lines:
        # Use original filename without extension
        base_name = os.path.splitext(file_name)[0]

        out_image_name = f"{base_name}.jpg"
        out_label_name = f"{base_name}.txt"

        bg.convert("RGB").save(os.path.join(OUTPUT_IMG_DIR, out_image_name))

        with open(os.path.join(OUTPUT_LABEL_DIR, out_label_name), "w") as f:
            f.write("\n".join(yolo_lines))

        counter += 1

print("Done generating synthetic YOLO images!")
