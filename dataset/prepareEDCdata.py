import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths to saved data
edc_folder_path = r"dataset/room_acoustic_parameters/EDC"

# Load EDC data: EDC data is stored in .npy files in the specified folder
npy_files = [f for f in os.listdir(edc_folder_path) if f.endswith('.npy')]
target_length = 44100  # Target length for EDC arrays

# List to store loaded EDC arrays
all_edcs = []

# Load each file
for fname in npy_files:
    full_path = os.path.join(edc_folder_path, fname)
    try:
        edc_array = np.load(full_path)

        # Flatten to 1D if needed
        if edc_array.ndim > 1:
            edc_array = edc_array.flatten()

        # Pad or trim to target length
        if len(edc_array) > target_length:
            edc_array = edc_array[:target_length]
        elif len(edc_array) < target_length:
            edc_array = np.pad(edc_array, (0, target_length - len(edc_array)), mode='constant')

        all_edcs.append(edc_array)
        print(f"Loaded {fname} with standardized shape: {edc_array.shape}")

    except Exception as e:
        print(f"Failed to load {fname}: {e}")

# Stack all EDCs into a single 2D numpy array
combined_edc_data = np.stack(all_edcs, axis=0)
print(f"\n Final combined EDC shape: {combined_edc_data.shape}")