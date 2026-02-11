import os
import cv2
import numpy as np
import random
from glob import glob

# Define paths
solar_dir = '/input/projects/Container/solar/test/images'
annot_dir = '/input/projects/Container/solar/test/labels'
container_dir = '/input/projects/Container/solar/container_backgrounds'
output_dir = 'synthetic_dataset/test'

os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

# Function to load YOLO label file. Structured as (cls, x, y, w, h)
def load_yolo_annotations(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls, x, y, w, h = map(float, parts[:5])
        boxes.append((cls, x, y, w, h))
    return boxes

# Function to convert YOLO format to pixel coordinates
def yolo_to_bbox(yolo_box, img_shape):
    cls, x, y, w, h = yolo_box
    H, W = img_shape[:2]
    x1 = int((x - w / 2) * W)
    y1 = int((y - h / 2) * H)
    x2 = int((x + w / 2) * W)
    y2 = int((y + h / 2) * H)
    return int(cls), x1, y1, x2, y2

# Function to convert pixel coordinates to YOLO format
def bbox_to_yolo(cls, x1, y1, x2, y2, img_shape):
    H, W = img_shape[:2]
    x = ((x1 + x2) / 2) / W
    y = ((y1 + y2) / 2) / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    return f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"

# Function to segment contaminant from background using Otsu-thresholding
def segment_contaminant(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    segmented = cv2.bitwise_and(crop, crop, mask=mask)
    return segmented, mask

# Paste segmented contaminant on background and calculate new bounding box
def paste_on_background(bg, obj, mask):
    h_bg, w_bg = bg.shape[:2]
    h_obj, w_obj = obj.shape[:2]

    # Scale randomly
    scale = random.uniform(0.4, 1.0)
    new_w = int(w_obj * scale)
    new_h = int(h_obj * scale)
    if new_w >= w_bg or new_h >= h_bg:
        scale = min(w_bg / w_obj * 0.5, h_bg / h_obj * 0.5)
        new_w = int(w_obj * scale)
        new_h = int(h_obj * scale)

    if new_w <= 0 or new_h <= 0:
        print(f"Skipping object due to invalid resize dimensions: ({new_w}, {new_h})")
        return bg, None 
    
    obj = cv2.resize(obj, (new_w, new_h))
    mask = cv2.resize(mask, (new_w, new_h))

    # Random position
    x_offset = random.randint(0, w_bg - new_w)
    y_offset = random.randint(0, h_bg - new_h)

    # Paste with mask
    roi = bg[y_offset:y_offset+new_h, x_offset:x_offset+new_w]
    bg[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = np.where(mask[..., None] == 255, obj, roi)

    return bg, (x_offset, y_offset, x_offset + new_w, y_offset + new_h)

# Find all images and backgrounds
solar_images = glob(os.path.join(solar_dir, "*.jpg"))
container_images = glob(os.path.join(container_dir, "*.jpg")) + \
                   glob(os.path.join(container_dir, "*.png"))

print(f"Found {len(solar_images)} solar images and {len(container_images)} container backgrounds")

# Go through all images
for solar_path in solar_images:
    base_name = os.path.splitext(os.path.basename(solar_path))[0]
    label_path = os.path.join(annot_dir, base_name + ".txt")
    if not os.path.exists(label_path):
        print(f"No label file for {base_name}")
        continue

    solar_img = cv2.imread(solar_path)
    if solar_img is None:
        print(f"Cannot read image {solar_path}")
        continue

    boxes = load_yolo_annotations(label_path)
    if len(boxes) == 0:
        print(f"No boxes found in {label_path}")
        continue

    bg_path = random.choice(container_images)
    bg = cv2.imread(bg_path)
    bg = cv2.resize(bg, (640, 640))

    label_lines = []

    for yolo_box in boxes:
        cls, x1, y1, x2, y2 = yolo_to_bbox(yolo_box, solar_img.shape)
        crop = solar_img[y1:y2, x1:x2]
        if crop.size == 0:
            print(f"Empty crop in {base_name}: {x1,y1,x2,y2}")
            continue

        obj, mask = segment_contaminant(crop)
        bg, bbox = paste_on_background(bg, obj, mask)
        
        if bbox is None:
            continue  # Skip adding this object to the label file
        
        bx1, by1, bx2, by2 = bbox
        label_line = bbox_to_yolo(cls, bx1, by1, bx2, by2, bg.shape)
        label_lines.append(label_line)

    if not label_lines:
        print(f"sNo valid objects in {base_name}, skipping save")
        continue

    out_img_path = os.path.join(output_dir, "images", f"{base_name}_synthetic.jpg")
    out_label_path = os.path.join(output_dir, "labels", f"{base_name}_synthetic.txt")

    cv2.imwrite(out_img_path, bg)
    with open(out_label_path, "w") as f:
        f.write("\n".join(label_lines))

    print(f"Saved synthetic: {out_img_path} with {len(label_lines)} objects")

print("\nDone generating synthetic dataset!")
