from ultralytics import YOLO

model = YOLO("/workspace/hrrcxh/yolov12/runs/detect/train22/weights/best.pt")

results = model.val(data="/input/projects/Container/soil_container/final_dataset/data.yaml", split="test", imgsz=640, plots=True)

print(results)
