import os
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ======================================================
# Paths & constants
# ======================================================
DATASET_DIR = "dataset/room_acoustic_largedataset/final_dataset"
MODEL_DIR = "Models"
RESULTS_DIR = "results"

FIG4_DIR = f"{RESULTS_DIR}/plots/fig4_edc"
FIG5_DIR = f"{RESULTS_DIR}/plots/fig5_rt30"
FIG6_DIR = f"{RESULTS_DIR}/plots/fig6_c50"
TABLE_DIR = f"{RESULTS_DIR}/tables"

for d in [FIG4_DIR, FIG5_DIR, FIG6_DIR, TABLE_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = "cpu"
EPS = 1e-8
N_EVAL = 100     # number of samples used for paper plots
SAMPLE_ID = 0    # for Fig. 4 (single room example)

# ======================================================
# Load dataset
# ======================================================
X = np.load(os.path.join(DATASET_DIR, "X_multiband.npy"))[-N_EVAL:]
Y = np.load(os.path.join(DATASET_DIR, "Y_multiband_downsampled.npy"))[-N_EVAL:]

X[~np.isfinite(X)] = 0
Y[~np.isfinite(Y)] = EPS

N, T, B = Y.shape

# ======================================================
# Load scalers
# ======================================================
scaler_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X_multiband.save"))
scaler_Y = joblib.load(os.path.join(MODEL_DIR, "scaler_Y_multiband.save"))

X_scaled = scaler_X.transform(X)

# ======================================================
# Model definition (same as training)
# ======================================================
class MultiBandLSTM(nn.Module):
    def __init__(self, input_size=16, hidden_size=128, num_layers=2, n_bands=7):
        super().__init__()
        self.fc_in = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc_out = nn.Linear(hidden_size, n_bands)

    def forward(self, x):
        x = self.fc_in(x)
        x = x.unsqueeze(1).repeat(1, T, 1)
        out, _ = self.lstm(x)
        return self.fc_out(out)

# ======================================================
# Load trained model
# ======================================================
ckpt = torch.load(os.path.join(MODEL_DIR, "multiband_lstm.ckpt"),
                  map_location=DEVICE)

model = MultiBandLSTM(n_bands=B)
model.load_state_dict(ckpt["model_state"])
model.eval()

# ======================================================
# Predict
# ======================================================
with torch.no_grad():
    Y_pred_scaled = model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()

Y_pred = scaler_Y.inverse_transform(
    Y_pred_scaled.reshape(N * T, B)
).reshape(N, T, B)

Y_pred = np.maximum(Y_pred, EPS)

# ======================================================
# Utility functions
# ======================================================
def edc_db(edc):
    return 10 * np.log10(np.maximum(edc, EPS))

def rt30_from_edc(edc):
    edc_db_vals = edc_db(edc)
    t = np.arange(len(edc_db_vals))
    valid = (edc_db_vals < -5) & (edc_db_vals > -35)
    if valid.sum() < 10:
        return np.nan
    slope, _ = np.polyfit(t[valid], edc_db_vals[valid], 1)
    return -30 / slope

def c50_from_edc(edc, early_ms=50):
    split = int(len(edc) * early_ms / 1000)
    return 10 * np.log10(
        np.sum(edc[:split]) / np.sum(edc[split:])
    )

# ======================================================
# FIGURE 4 — EDC (GT vs Pred)
# ======================================================
plt.figure(figsize=(6, 4))

for b in range(B):
    plt.plot(edc_db(Y[SAMPLE_ID, :, b]), linewidth=2)

plt.plot(edc_db(Y_pred[SAMPLE_ID, :, 0]),
         "--", linewidth=2, color="black", label="Predicted")

plt.title("Target and Predicted EDCs (Random Room)")
plt.xlabel("Time Frames")
plt.ylabel("Energy (dB)")
plt.grid(True)
plt.tight_layout()

plt.savefig(f"{FIG4_DIR}/fig4_edc_gt_vs_pred.png", dpi=300)
plt.close()

# ======================================================
# FIGURE 5 — RT30 scatter (GT vs Pred)
# ======================================================
rt30_gt = []
rt30_pr = []

for i in range(N):
    for b in range(B):
        rt30_gt.append(rt30_from_edc(Y[i, :, b]))
        rt30_pr.append(rt30_from_edc(Y_pred[i, :, b]))

rt30_gt = np.array(rt30_gt)
rt30_pr = np.array(rt30_pr)

plt.figure(figsize=(5, 5))
plt.scatter(rt30_gt, rt30_pr, alpha=0.6)
lims = [0, max(np.nanmax(rt30_gt), np.nanmax(rt30_pr))]
plt.plot(lims, lims, "--", color="gray")

plt.xlabel("Target RT30 (s)")
plt.ylabel("Predicted RT30 (s)")
plt.title("RT30 Comparison")
plt.grid(True)
plt.tight_layout()

plt.savefig(f"{FIG5_DIR}/fig5_rt30_scatter.png", dpi=300)
plt.close()

# ======================================================
# FIGURE 6 — C50 scatter (GT vs Pred)
# ======================================================
c50_gt = []
c50_pr = []

for i in range(N):
    for b in range(B):
        c50_gt.append(c50_from_edc(Y[i, :, b]))
        c50_pr.append(c50_from_edc(Y_pred[i, :, b]))

c50_gt = np.array(c50_gt)
c50_pr = np.array(c50_pr)

plt.figure(figsize=(5, 5))
plt.scatter(c50_gt, c50_pr, alpha=0.6)
lims = [min(c50_gt.min(), c50_pr.min()),
        max(c50_gt.max(), c50_pr.max())]
plt.plot(lims, lims, "--", color="gray")

plt.xlabel("Target C50 (dB)")
plt.ylabel("Predicted C50 (dB)")
plt.title("C50 Comparison")
plt.grid(True)
plt.tight_layout()

plt.savefig(f"{FIG6_DIR}/fig6_c50_scatter.png", dpi=300)
plt.close()

# ======================================================
# TABLE 2 — Metrics
# ======================================================
metrics = {
    "Metric": ["RT30 (s)", "C50 (dB)"],
    "MAE": [
        mean_absolute_error(rt30_gt, rt30_pr),
        mean_absolute_error(c50_gt, c50_pr)
    ],
    "RMSE": [
        mean_squared_error(rt30_gt, rt30_pr, squared=False),
        mean_squared_error(c50_gt, c50_pr, squared=False)
    ],
    "R2": [
        r2_score(rt30_gt, rt30_pr),
        r2_score(c50_gt, c50_pr)
    ]
}

df = pd.DataFrame(metrics)
df.to_csv(f"{TABLE_DIR}/table2_metrics.csv", index=False)

print("✅ Fig. 4 saved:", FIG4_DIR)
print("✅ Fig. 5 saved:", FIG5_DIR)
print("✅ Fig. 6 saved:", FIG6_DIR)
print("✅ Table 2 saved:", TABLE_DIR)
