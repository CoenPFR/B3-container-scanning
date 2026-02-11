from ultralytics import YOLO
model = YOLO('/workspace/hrrcxh/yolov12/runs/detect/train26/weights/best.pt') # YOLO('yolov12x.pt')  # YOLO('yolov12x.yaml') # YOLO('/workspace/hrrcxh/yolov12/runs/detect/train7/weights/best.pt') 
# results = model.train(resume=True)
results = model.train(data="/input/projects/Container/soil_container/final_dataset/data.yaml",
                      epochs=200,
                      batch=16,
                      imgsz=640,
                      lr0=0.005
                     )

