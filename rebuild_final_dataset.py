import os
import numpy as np
import pandas as pd
from tqdm import tqdm

BASE = "dataset/room_acoustic_largedataset/final_dataset"
X_DIR = os.path.join(BASE, "X")
Y_DIR = os.path.join(BASE, "Y")
INDEX = os.path.join(BASE, "index.csv")

df = pd.read_csv(INDEX)

X_list, ids = [], []

for _, row in tqdm(df.iterrows(), total=len(df)):
    rid = row["id"]

    x_path = os.path.join(X_DIR, f"{rid}.npy")
    y_path = os.path.join(Y_DIR, f"{rid}.npy")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        continue

    y = np.load(y_path)
    if not np.isfinite(y).any():
        continue

    X_list.append(np.load(x_path))
    ids.append(rid)

X = np.stack(X_list)

np.save(os.path.join(BASE, "X_multiband.npy"), X)
np.save(os.path.join(BASE, "ids.npy"), np.array(ids))

print("✅ Dataset rebuilt safely")
print("X:", X.shape)
print("Samples:", len(ids))
