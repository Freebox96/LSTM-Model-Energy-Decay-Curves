import os
import numpy as np
from tqdm import tqdm

DATASET_DIR = "dataset/room_acoustic_largedataset/final_dataset/Y"
OUTPUT_FILE = "dataset/room_acoustic_largedataset/final_dataset/Y_multiband_downsampled.npy"
DOWNSAMPLE_FACTOR = 100  # 144000 -> 1440

# List all Y files
y_files = sorted(os.listdir(DATASET_DIR))
Y_down_list = []

print(f"Found {len(y_files)} Y files, processing one by one...")

for f in tqdm(y_files):
    y_path = os.path.join(DATASET_DIR, f)
    y = np.load(y_path)  # shape: (T, B)
    
    # Downsample along time axis
    y_down = y[::DOWNSAMPLE_FACTOR, :]
    
    Y_down_list.append(y_down)

# Stack into single array
Y_down = np.stack(Y_down_list)  # shape: (N, T_down, B)
print("Downsampled Y shape:", Y_down.shape)

# Save
np.save(OUTPUT_FILE, Y_down)
print(f"✅ Downsampled Y saved to {OUTPUT_FILE}")
