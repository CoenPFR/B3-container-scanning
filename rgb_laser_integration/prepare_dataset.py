import os
import numpy as np
import cv2
import pandas as pd
import tifffile as tiff
import glob

# Homography camera to laser (copied from Obtain_Scaling_and_Alignment.ipynb)
H_c2l = np.array([[ 1.57666337e+00, -4.73290772e-02, -4.82136439e+02],
                  [ 2.19054968e-02,  1.53502952e+00, -1.86576837e+02],
                  [ 5.08617971e-05, -8.19856335e-05,  1.00000000e+00]])

# Directories
data_folder = r"K:\ALL\coen\laser\exp1"
output_clean = r"K:\ALL\coen\laser\exp1\output\clean"
output_soiled = r"K:\ALL\coen\laser\exp1\output\soiled"
os.makedirs(output_clean, exist_ok=True)
os.makedirs(output_soiled, exist_ok=True)

# Load tile annotation CSV (soiled/not soiled)
tile_csv = r"K:\ALL\coen\laser\exp1\tile_annotation.csv"
tile_mapping = pd.read_csv(tile_csv, index_col=0)  # name column as index

# Function to crop ROI and apply scaling factor to laser images (scaling factor copied from Obtain_Scaling_and_Alignment.ipynb)
def preprocess_laser(laser_img, col_start=4200, col_end=7100, laser_scale=0.24885201):
    img_crop = laser_img[:, col_start:col_end]
    rows, cols = img_crop.shape
    new_cols = int(cols * laser_scale)
    img_rescale = cv2.resize(img_crop, (new_cols, rows), interpolation=cv2.INTER_AREA)
    return img_rescale

# Function to apply perspective transformation (homography)
def align_images(img1, img2, H):
    img2_height, img2_width = img2.shape[:2]
    aligned_img1 = cv2.warpPerspective(img1, H, (img2_width, img2_height))
    return aligned_img1

# Fuction to split image into four equal tiles
def split_into_tiles(img):
    # Crop to even dimensions to ensure equal output size
    img_cropped = img[:828, :720, :]
    H, W, C = img_cropped.shape
    h_mid, w_mid = H//2, W//2
    return [
        img_cropped[:h_mid, :w_mid],
        img_cropped[:h_mid, w_mid:],
        img_cropped[h_mid:, :w_mid],
        img_cropped[h_mid:, w_mid:]
    ]

# Loop through all samples
for idx in range(1, 21):  # sample1 - sample20
    print(f"processing sample {idx}")
    sample_name = f"sample{idx}"
    
    # Camera path
    camera_path = os.path.join(data_folder, f"{sample_name}.jpg")

    # Laser paths
    laser1270_file = glob.glob(os.path.join(data_folder, f"{sample_name}_laser_1270nm_Dev1_ai1_*.csv"))[0]
    laser1650_file = glob.glob(os.path.join(data_folder, f"{sample_name}_laser_1650nm_Dev1_ai0_*.csv"))[0]

    # Load images
    camera_img = cv2.imread(camera_path)
    laser1270_raw = pd.read_csv(laser1270_file, delimiter=",", skiprows=4).to_numpy()
    laser1270_raw = np.flipud(laser1270_raw)
    laser1650_raw = pd.read_csv(laser1650_file, delimiter=",", skiprows=4).to_numpy()
    laser1650_raw = np.flipud(laser1650_raw)

    # Preprocess laser images
    laser1270 = preprocess_laser(laser1270_raw)
    laser1650 = preprocess_laser(laser1650_raw)

    # Align camera to laser1270
    aligned_camera = align_images(camera_img, laser1270, H_c2l)

    # Stack channels: camera RGB + laser1270 + laser1650
    laser1270 = laser1270[:, :, None]
    laser1650 = laser1650[:, :, None]
    stacked_img = np.concatenate([aligned_camera.astype(np.float32), laser1270.astype(np.float32), laser1650.astype(np.float32)], axis=2)

    # Split into 4 tiles
    tiles = split_into_tiles(stacked_img)

    # Get clean tile index from CSV
    clean_tile_idx = tile_mapping.loc[sample_name, "clean_tile"]

    # Save tiles
    for t_idx, tile in enumerate(tiles):
        cls_folder = output_clean if t_idx == clean_tile_idx else output_soiled
        save_path = os.path.join(cls_folder, f"{sample_name}_tile{t_idx}.tiff")
        tiff.imwrite(save_path, tile.astype(np.float32))
