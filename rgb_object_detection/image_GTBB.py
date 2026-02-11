import os
import cv2

# paths
image_dir = "/input/projects/Container/stone_synthetic/visualized_bboxes"
label_dir = "/input/projects/Container/stone_synthetic/output/val/labels"
output_dir = "/input/projects/Container/stone_synthetic/visualized_bboxes/result"

os.makedirs(output_dir, exist_ok=True)

# class name(s)
class_names = ["soil"]

# loop
for img_name in os.listdir(image_dir):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(image_dir, img_name)
    label_path = os.path.join(
        label_dir, os.path.splitext(img_name)[0] + ".txt"
    )

    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read {img_path}")
        continue

    h, w = img.shape[:2]

    if not os.path.exists(label_path):
        print(f"No label for {img_name}")
        cv2.imwrite(os.path.join(output_dir, img_name), img)
        continue

    with open(label_path) as f:
        for line in f:
            cls, cx, cy, bw, bh = map(float, line.split())

            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            # draw bounding box (green)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # # label
            # label = class_names[int(cls)]
            # cv2.putText(
            #     img,
            #     label,
            #     (x1, max(y1 - 5, 15)),
            #     cv2.FONT_HERSHEY_DUPLEX,
            #     0.5,
            #     (0, 255, 0),
            #     1
            # )

    cv2.imwrite(os.path.join(output_dir, img_name), img)

print("Image with GT bounding boxes saved to:", output_dir)

