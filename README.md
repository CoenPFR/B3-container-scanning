This repository contains work from the internship of Coen Hakvoort.
Three different components are documented:
1. YOLOv12-based RGB object detection of soil contaminants. Pretraining on synthetic datasets, generated through copy-paste augmentation was evaluated. 
2. Wavelength selection for the laser scanner. Ranking of hyperspectral bands based on the Fisher-score (class separability) was performed for five classes: soil, insect, leaf, rust, container.
3. RGB-Laser integration experiment: RGB and laser images were captured and fused by channel stacking, and used for binary classification of soiled and clean container patches, using a modified MobileNetV3 architecture.
