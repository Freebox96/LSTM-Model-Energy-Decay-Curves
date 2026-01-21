import os
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# Paths
# ===============================
DATASET_DIR = "dataset/room_acoustic_largedataset/final_dataset"
MODEL_DIR = "Models"
RESULTS_DIR = "results"

PLOT_DIR = f"{RESULTS_DIR}/plots"
TABLE_DIR = f"{RESULTS_DIR}/tables"

os.makedirs(f"{PLOT_DIR}/edc", exist_ok=True)
os.makedirs(f"{PLOT_DIR}/c50", exist_ok=True)
os.makedirs(f"{PLOT_DIR}/spectral", exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

DEVICE = "cpu"
N_EVAL = 50
EARLY_MS = 50
EPS = 1e-10

# ===============================
# Load data
# ===============================
X = np.load(os.path.join(DATASET_DIR, "X_multiband.npy"))[-N_EVAL:]
Y = np.load(os.path.join(DATASET_DIR, "Y_multiband_downsampled.npy"))[-N_EVAL:]

X[~np.isfinite(X)] = 0.0
Y[~np.isfinite(Y)] = EPS
Y = np.maximum(Y, EPS)

# ===============================
# Load scalers
# ===============================
scaler_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X_multiband.save"))
scaler_Y = joblib.load(os.path.join(MODEL_DIR, "scaler_Y_multiband.save"))
X_scaled = scaler_X.transform(X)

# ===============================
# Model
# ===============================
class MultiBandLSTM(nn.Module):
    def __init__(self, input_size=16, hidden_size=128, num_layers=2, n_bands=7):
        super().__init__()
        self.fc_in = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_size, n_bands)

    def forward(self, x):
        x = self.fc_in(x)
        x = x.unsqueeze(1).repeat(1, Y.shape[1], 1)
        out, _ = self.lstm(x)
        return self.fc_out(out)

# ===============================
# Load model (NO TRAINING)
# ===============================
ckpt = torch.load(os.path.join(MODEL_DIR, "multiband_lstm.ckpt"),
                  map_location=DEVICE)

model = MultiBandLSTM(n_bands=Y.shape[2])
model.load_state_dict(ckpt["model_state"])
model.eval()

# ===============================
# Predict
# ===============================
with torch.no_grad():
    Y_pred_scaled = model(
        torch.tensor(X_scaled, dtype=torch.float32)
    ).numpy()

N, T, B = Y_pred_scaled.shape

Y_pred = scaler_Y.inverse_transform(
    Y_pred_scaled.reshape(N * T, B)
).reshape(N, T, B)

Y_pred = np.maximum(Y_pred, EPS)

# ===============================
# Metrics (NUMERICALLY SAFE)
# ===============================
def energy_ratio_early_late(edc):
    edc = np.maximum(edc, EPS)

    split = max(1, int(T * EARLY_MS / 1000))
    early = np.sum(edc[:split])
    late = np.sum(edc[split:])

    if late <= 0:
        return np.nan

    return 10 * np.log10(early / late)

def decay_slope_rt(edc):
    edc = np.maximum(edc, EPS)
    edc_db = 10 * np.log10(edc)

    t = np.arange(len(edc_db))
    valid = (edc_db < -5) & (edc_db > -35)

    if valid.sum() < 10:
        return np.nan

    slope, _ = np.polyfit(t[valid], edc_db[valid], 1)
    return -30 / slope if slope < 0 else np.nan

def spectral_similarity(gt, pr):
    gt = np.maximum(gt, EPS)
    pr = np.maximum(pr, EPS)

    if np.all(gt == 0) or np.all(pr == 0):
        return np.nan

    return cosine_similarity(
        gt.reshape(1, -1),
        pr.reshape(1, -1)
    )[0, 0]

# ===============================
# 1️⃣ EDC PLOTS
# ===============================
sample_id = 0

for b in range(B):
    gt = Y[sample_id, :, b]
    pr = Y_pred[sample_id, :, b]

    gt_db = 10 * np.log10(np.maximum(gt, EPS))
    pr_db = 10 * np.log10(np.maximum(pr, EPS))

    plt.figure(figsize=(6, 4))
    plt.plot(gt_db, label="GT", linewidth=2)
    plt.plot(pr_db, "--", label="Pred")

    plt.title(f"EDC – Band {b}")
    plt.xlabel("Time frames")
    plt.ylabel("Energy (dB)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"{PLOT_DIR}/edc/edc_band_{b}.png", dpi=300)
    plt.close()

# ===============================
# 2️⃣ TABLES
# ===============================
rows = []

for b in range(B):
    for i in range(N):
        rows.append({
            "Band": b,
            "RT30_GT": decay_slope_rt(Y[i, :, b]),
            "RT30_PRED": decay_slope_rt(Y_pred[i, :, b]),
            "C50_GT": energy_ratio_early_late(Y[i, :, b]),
            "C50_PRED": energy_ratio_early_late(Y_pred[i, :, b]),
            "Spectral_Similarity": spectral_similarity(
                Y[i, :, b],
                Y_pred[i, :, b]
            )
        })

df = pd.DataFrame(rows)

df.groupby("Band").mean().to_csv(
    f"{TABLE_DIR}/multiband_metrics.csv"
)

print("✅ EDC plots saved to results/plots/edc/")
print("✅ Tables saved to results/tables/multiband_metrics.csv")
