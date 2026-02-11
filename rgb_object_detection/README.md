# RGB Detection of soil contaminants on containers

## YOLOv12 object detection model

### Installation
Follow instructions from: https://github.com/sunsmarterjie/yolov12

### Data
Required YOLO dataset structure:
```md
    -dataset 	
	|-train	
	|   |-images
	|	|	|-image1.jpg
	|	|	|-image2.jpg
	|	|	|-...
	|   |-labels
	|		|-image1.txt
	|		|-image2.txt
	|		|-...
	|
	|-val
	|   |-images
	|   |-labels
	|
	|-test
	|   |-images
	|   |-labels
	|
	|-data.yaml
```
Datasets used during this internship are stored at the following locations:
- final dataset with soil contaminants:
```md
\\storage.powerplant.pfr.co.nz\input\projects\Container\soil_container\final_dataset
```
- solar synthetic dataset:
```md
\\storage.powerplant.pfr.co.nz\input\projects\Container\solar\synthetic_dataset
```
- stone synthetic dataset:
```md
\\storage.powerplant.pfr.co.nz\input\projects\Container\stone_synthetic\output
```

### Training
Modify `train_model.py` -> fill in correct model, data.yaml and training parameters  
Modify `yolov12_train.sl` -> for more information about job scripts and submission: https://wiki.powerplant.pfr.co.nz/Slurm/faq

In Putty terminal:
```bash
# Submit job
srun yolov12_train.sl
# Check job status
squeue -u $USER
# Follow progress of script
tail -f yolov12_train.err
tail -f yolov12_train.out
```

Trained weights are stored in `\runs\detect\train..\weights\best.pt`

### Evaluation
Modify `val_model.py` -> fill in trained model and correct data.yaml  
Modify `yolov12_val.sl`

In Putty terminal:
```bash
srun val_model
```

Evaluation metrics are stored in `yolov12_val.out`, and `\runs\detect\val..`

## Synthetic data
The solar synthetic dataset is created using `create_solar_synt.py`  
The stone synthetic dataset is created using `create_stone_synt.py`

## Plot images with bounding boxes
Use `image_GTBB.py`




