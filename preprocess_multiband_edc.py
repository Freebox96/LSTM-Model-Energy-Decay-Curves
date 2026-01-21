import os
import numpy as np
import pandas as pd

# Paths
CSV_PATH = "dataset/room_acoustic_largedataset/roomFeaturesDataset.csv"
EDC_MB_PATH = "dataset/room_acoustic_largedataset/EDC_multiband"
OUT_PATH = "dataset/room_acoustic_largedataset/EDC_multiband_processed"

# Create output folder if it doesn't exist
os.makedirs(OUT_PATH, exist_ok=True)

# Load CSV
df = pd.read_csv(CSV_PATH)
ids = df["ID"].tolist()  # List of IDs like 'RIR_001_case0_edc'

print(f"Processing {len(ids)} RIR IDs from CSV...")

matched_count = 0
skipped_files = []

for csv_id in ids:
    # Map CSV ID to actual multiband file
    mb_filename = csv_id.replace("_edc", "_multiband") + ".npy"
    mb_filepath = os.path.join(EDC_MB_PATH, mb_filename)
    
    if not os.path.isfile(mb_filepath):
        print(f"⚠️  File not found for ID {csv_id}: {mb_filename}")
        skipped_files.append(csv_id)
        continue

    # Load the multiband file
    try:
        multiband = np.load(mb_filepath)
        if multiband.size == 0:
            print(f"⚠️  Empty array for ID {csv_id}: {mb_filename}")
            skipped_files.append(csv_id)
            continue
        
        # Save processed array
        out_name = csv_id + "_processed.npy"
        out_filepath = os.path.join(OUT_PATH, out_name)
        np.save(out_filepath, multiband)
        matched_count += 1

    except Exception as e:
        print(f"❌ Failed to process {csv_id}: {e}")
        skipped_files.append(csv_id)

print("\n✅ Processing complete!")
print(f"Matched and saved: {matched_count}")
print(f"Skipped files: {len(skipped_files)}")
if skipped_files:
    print(skipped_files)
