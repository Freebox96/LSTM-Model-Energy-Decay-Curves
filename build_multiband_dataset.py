import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# ===============================
# Paths
# ===============================
BASE_PATH = "dataset/room_acoustic_largedataset"
FEATURES_CSV = os.path.join(BASE_PATH, "roomFeaturesDataset.csv")
EDC_MB_PATH = os.path.join(BASE_PATH, "EDC_multiband")
OUT_PATH = os.path.join(BASE_PATH, "final_dataset")

X_OUT = os.path.join(OUT_PATH, "X")
Y_OUT = os.path.join(OUT_PATH, "Y")

os.makedirs(X_OUT, exist_ok=True)
os.makedirs(Y_OUT, exist_ok=True)

# ===============================
# Load features
# ===============================
df = pd.read_csv(FEATURES_CSV)

ids = df["ID"].values
features = df.drop(columns=["ID"]).values.astype(np.float32)

print("Loaded features:", features.shape)

matched = 0
index_rows = []

# ===============================
# Build dataset (SAFE)
# ===============================
for i, room_id in enumerate(tqdm(ids)):
    base = room_id.replace("_edc", "")
    mb_file = f"{base}_multiband.npy"
    mb_path = os.path.join(EDC_MB_PATH, mb_file)

    if not os.path.exists(mb_path):
        continue

    try:
        y = np.load(mb_path).astype(np.float32)
        x = features[i]

        # Save individually (NO stacking)
        np.save(os.path.join(X_OUT, f"{base}.npy"), x)
        np.save(os.path.join(Y_OUT, f"{base}.npy"), y)

        index_rows.append(base)
        matched += 1

    except Exception as e:
        print(f"❌ Failed {base}: {e}")
        continue

# ===============================
# Save index
# ===============================
pd.DataFrame({"id": index_rows}).to_csv(
    os.path.join(OUT_PATH, "index.csv"), index=False
)

print("Matched samples:", matched)
print("✅ Final dataset saved safely to:", OUT_PATH)
