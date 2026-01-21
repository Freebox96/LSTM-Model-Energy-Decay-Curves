import os
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# Config
# ===============================
DATASET_DIR = "dataset/room_acoustic_largedataset/final_dataset"
MODEL_DIR = "Models"
DEVICE = "cpu"

N_EVAL = 50
EARLY_MS = 50
FS = 48000  # relative only (EDC domain)

# ===============================
# Load dataset (reuse training data)
# ===============================
X_all = np.load(os.path.join(DATASET_DIR, "X_multiband.npy"))
Y_all = np.load(os.path.join(DATASET_DIR, "Y_multiband_downsampled.npy"))

X = X_all[-N_EVAL:]
Y = Y_all[-N_EVAL:]

print("Eval X:", X.shape)
print("Eval Y:", Y.shape)

X = np.nan_to_num(X)
Y = np.nan_to_num(Y)

# ===============================
# Load scalers
# ===============================
scaler_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X_multiband.save"))
scaler_Y = joblib.load(os.path.join(MODEL_DIR, "scaler_Y_multiband.save"))

X_scaled = scaler_X.transform(X)

# ===============================
# Model definition (IDENTICAL to training)
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
# Load trained model (NO TRAINING)
# ===============================
ckpt = torch.load(
    os.path.join(MODEL_DIR, "multiband_lstm.ckpt"),
    map_location=DEVICE
)

model = MultiBandLSTM(n_bands=Y.shape[2]).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

# ===============================
# Predict
# ===============================
with torch.no_grad():
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    Y_pred_scaled = model(X_tensor).cpu().numpy()

# inverse scaling
N, T, B = Y_pred_scaled.shape
Y_pred = scaler_Y.inverse_transform(
    Y_pred_scaled.reshape(N * T, B)
).reshape(N, T, B)

Y_pred = np.maximum(Y_pred, 1e-10)

# ===============================
# Metric functions
# ===============================
def decay_slope_rt(edc):
    """RT30 proxy via decay slope"""
    edc_db = 10 * np.log10(np.maximum(edc, 1e-10))
    t = np.arange(len(edc_db)) / FS

    valid = (edc_db < -5) & (edc_db > -35)
    if np.sum(valid) < 10:
        return np.nan

    slope, _ = np.polyfit(t[valid], edc_db[valid], 1)
    return -30 / slope if slope < 0 else np.nan


def energy_ratio_early_late(edc, early_ms=EARLY_MS):
    """C50-compatible clarity proxy"""
    split = int(len(edc) * early_ms / 1000)
    early = np.sum(edc[:split])
    late = np.sum(edc[split:])
    return 10 * np.log10(early / late) if late > 0 else np.nan


def spectral_similarity(gt, pred):
    gt = gt.reshape(1, -1)
    pred = pred.reshape(1, -1)
    return cosine_similarity(gt, pred)[0, 0]

# ===============================
# Evaluation
# ===============================
print("\n===== MULTI-BAND EDC EVALUATION =====")

for b in range(B):
    rt_gt, rt_pr = [], []
    c_gt, c_pr = [], []
    spec = []

    for i in range(N):
        gt = Y[i, :, b]
        pr = Y_pred[i, :, b]

        if np.any(gt <= 0) or np.any(pr <= 0):
            continue

        rt_gt.append(decay_slope_rt(gt))
        rt_pr.append(decay_slope_rt(pr))

        c_gt.append(energy_ratio_early_late(gt))
        c_pr.append(energy_ratio_early_late(pr))

        spec.append(spectral_similarity(gt, pr))

    print(f"\nBand {b}")

    if len(spec) == 0 or np.all(np.isnan(spec)):
        print("⚠️  Band not evaluable")
        continue

    rt_gt_m = np.nanmean(rt_gt)
    rt_pr_m = np.nanmean(rt_pr)
    c_gt_m = np.nanmean(c_gt)
    c_pr_m = np.nanmean(c_pr)
    spec_m = np.nanmean(spec)

    improvement = 100 * (c_pr_m - c_gt_m) / abs(c_gt_m) if abs(c_gt_m) > 1e-6 else 0

    print(f"RT30_GT:   {rt_gt_m:.2f}")
    print(f"RT30_PRED: {rt_pr_m:.2f}")
    print(f"C50_GT:    {c_gt_m:.2f}")
    print(f"C50_PRED:  {c_pr_m:.2f}")
    print(f"Spectral_Sim: {spec_m:.4f}")

    if improvement >= 5:
        print(f"✅ ≥5% C50 improvement: {improvement:.2f}%")
    else:
        print(f"ℹ️  C50 change: {improvement:.2f}%")
