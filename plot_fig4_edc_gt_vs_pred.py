import os
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt

# ===============================
# Paths
# ===============================
DATASET_DIR = "dataset/room_acoustic_largedataset/final_dataset"
MODEL_DIR = "Models"
RESULTS_DIR = "results"
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(PLOT_DIR, exist_ok=True)

DEVICE = "cpu"
EPS = 1e-8

# ===============================
# Load data
# ===============================
X = np.load(os.path.join(DATASET_DIR, "X_multiband.npy"))
Y = np.load(os.path.join(DATASET_DIR, "Y_multiband_downsampled.npy"))

X[~np.isfinite(X)] = 0
Y[~np.isfinite(Y)] = EPS

# Random room (as in paper)
sample_id = np.random.randint(0, X.shape[0])

# ===============================
# Load scalers
# ===============================
scaler_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X_multiband.save"))
scaler_Y = joblib.load(os.path.join(MODEL_DIR, "scaler_Y_multiband.save"))

X_scaled = scaler_X.transform(X[[sample_id]])

# ===============================
# Model definition
# ===============================
class MultiBandLSTM(nn.Module):
    def __init__(self, input_size=16, hidden_size=128, num_layers=2, n_bands=7):
        super().__init__()
        self.fc_in = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
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
# Load trained model
# ===============================
ckpt = torch.load(
    os.path.join(MODEL_DIR, "multiband_lstm.ckpt"),
    map_location=DEVICE
)

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

T, B = Y_pred_scaled.shape[1], Y_pred_scaled.shape[2]

Y_pred = scaler_Y.inverse_transform(
    Y_pred_scaled.reshape(T, B)
).reshape(T, B)

Y_pred = np.maximum(Y_pred, EPS)

# ===============================
# FIG. 4 — Target vs Predicted EDCs
# ===============================
plt.figure(figsize=(9, 6))

# Ground Truth EDCs (solid, colored)
for b in range(B):
    plt.plot(
        10 * np.log10(Y[sample_id, :, b]),
        linewidth=2,
        alpha=0.9
    )

# Predicted EDC (single dashed black curve)
plt.plot(
    10 * np.log10(Y_pred[:, 0]),
    linestyle="--",
    color="black",
    linewidth=2.5
)

# Decay fitting region
plt.axhline(-5, color="black", linestyle="--", linewidth=1)
plt.axhline(-35, color="black", linestyle="--", linewidth=1)

# Axes formatting
plt.xlabel("Time Frames", fontsize=12)
plt.ylabel("Energy (dB)", fontsize=12)
plt.title(
    "Target and Predicted EDCs (Randomly Selected Room)",
    fontsize=14
)
plt.grid(True, linestyle="--", alpha=0.4)

# ===============================
# Small in-plot description (top-right, paper-style)
# ===============================
plt.text(
    0.98,
    0.98,
    "Solid colored lines: Ground-truth EDCs (bands)\n"
    "Black dashed line: Predicted EDC\n"
    "Horizontal dashed lines: Decay fit range",
    transform=plt.gca().transAxes,
    fontsize=9,               # tiny like the paper
    ha="right",
    va="top",
    bbox=dict(
        facecolor="white",
        edgecolor="none",
        alpha=0.85
    )
)

plt.tight_layout()

# ===============================
# Save
# ===============================
out_path = os.path.join(PLOT_DIR, "fig4_edc_gt_vs_pred.png")
plt.savefig(out_path, dpi=300)
plt.close()

print(f"✅ Fig. 4 saved to: {out_path}")
