# RGB-Laser integration experiment
## Prepare dataset
- Obtain the scaling factor for rescaling the laser images. Capture laser image of checkerboard, and use this code: `Obtain_Scaling_and_Alignment.ipynb`
- Obtain the extrinsic transformation (homography) between RGB camera and laser. Capture camera and laser image of checkerboard, and use the same notebook: `Obtain_Scaling_and_Alignment.ipynb`
- Prepare the dataset for classification using `prepare_dataset.py`:
    - Copy the extrinisc transformation (H_c2l) and rescaling factor from the previous steps to `prepare_dataset.py`. 
    - Running the file loads the camera and laser images, applies rescaling factor to laser channels, aligns camera image to laser, stacks channels to create 5-channel image, splits each image in four tiles and saves them in 'clean' or 'soiled' folder based on the annotations stored in `tile_annotation.csv`.

The processed tiles used in the experiment are stored at: `"K:\ALL\coen\laser\exp1\output"`.

## Perform 5-fold cross-validation of MobilenetV3 classification using RGB + laser and RGB-only data
Run `Mobilenet_Classification.ipnyb`

## Perform 5-fold cross-validation of linear SVC (not in report)
Classification performance of a linear SVC was performed on RGB + laser, RGB-only, and laser-only. Features were manually extracted and used with the linear SVC. Because it learns very few parameters compared to deep-learning models like MobileNetV3, it usually generalizes better to small datasets.  

Results show that the linear SVC also struggles to gain useful information from the laser channels, confirming that the signal-to-noise ratio is too low. 

Code: `ML_classification.ipynb`
