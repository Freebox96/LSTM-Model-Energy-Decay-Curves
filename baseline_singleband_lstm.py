import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib

# ===============================
# Config
# ===============================
DATASET_DIR = "dataset/room_acoustic_largedataset/final_dataset"
MODEL_DIR = "Models"
BAND_INDEX = 0          # single-band baseline
DEVICE = "cpu"          # Mac / CPU safe
EPS = 1e-8

os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================
# Load dataset (DOWN-SAMPLED Y)
# ===============================
X = np.load(os.path.join(DATASET_DIR, "X_multiband.npy"))  
Y = np.load(os.path.join(DATASET_DIR, "Y_multiband_downsampled.npy"))

print("X shape:", X.shape)      # (N, 16)
print("Y shape:", Y.shape)      # (N, T, B)

# ===============================
# Select single band
# ===============================
Y = Y[:, :, BAND_INDEX]     # (N, T)
Y = Y[:, :, None]           # (N, T, 1)

# ===============================
# 🔥 LOG-ENERGY COMPRESSION (CRITICAL)
# ===============================
Y = np.log(Y + EPS)

# ===============================
# Safety checks
# ===============================
assert np.isfinite(X).all(), "Non-finite values in X!"
assert np.isfinite(Y).all(), "Non-finite values in Y after log!"

# ===============================
# Scaling
# ===============================
scaler_X = StandardScaler()
scaler_Y = RobustScaler()     # safer for decay curves

X_scaled = scaler_X.fit_transform(X)

N, T, C = Y.shape
Y_flat = Y.reshape(N * T, C)
Y_scaled = scaler_Y.fit_transform(Y_flat).reshape(N, T, C)

joblib.dump(scaler_X, os.path.join(MODEL_DIR, "scaler_X.save"))
joblib.dump(scaler_Y, os.path.join(MODEL_DIR, "scaler_Y.save"))

# ===============================
# Torch tensors
# ===============================
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
Y_tensor = torch.tensor(Y_scaled, dtype=torch.float32)

# ===============================
# Model definition
# ===============================
class BaselineLSTM(nn.Module):
    def __init__(self, input_size=16, hidden_size=128, num_layers=2):
        super().__init__()
        self.fc_in = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x, seq_len):
        """
        x: (B, 16)
        returns: (B, T, 1)
        """
        x = self.fc_in(x)              # (B, H)
        x = x.unsqueeze(1)             # (B, 1, H)
        x = x.repeat(1, seq_len, 1)    # (B, T, H)
        out, _ = self.lstm(x)
        out = self.fc_out(out)
        return out

# ===============================
# Exports for training
# ===============================
__all__ = [
    "BaselineLSTM",
    "X_tensor",
    "Y_tensor",
    "DEVICE",
    "MODEL_DIR"
]
