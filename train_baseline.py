# ===============================
# train_baseline.py
# Baseline Single-Band LSTM (CPU / Mac safe)
# ===============================

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

# ===============================
# Config
# ===============================
DATASET_DIR = "dataset/room_acoustic_largedataset/final_dataset"
MODEL_DIR = "Models"

BAND_INDEX = 0
EPOCHS = 20
BATCH_SIZE = 16
LR = 1e-3
DEVICE = "cpu"

HIDDEN_SIZE = 96      # slightly lighter than 128
NUM_LAYERS = 2

os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================
# Load dataset
# ===============================
X = np.load(os.path.join(DATASET_DIR, "X_multiband.npy"))          # (N, 16)
Y = np.load(os.path.join(DATASET_DIR, "Y_multiband_downsampled.npy"))  # (N, T, B)

# Select single band
Y = Y[:, :, BAND_INDEX][:, :, None]  # (N, T, 1)

print("X shape:", X.shape)
print("Y shape:", Y.shape)

# ===============================
# Safety: NaN / Inf
# ===============================
X = np.nan_to_num(X)
Y = np.nan_to_num(Y)

# ===============================
# Scaling
# ===============================
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)

N, T, C = Y.shape
Y_flat = Y.reshape(N * T, C)
Y_scaled = scaler_Y.fit_transform(Y_flat).reshape(N, T, C)

joblib.dump(scaler_X, os.path.join(MODEL_DIR, "scaler_X.save"))
joblib.dump(scaler_Y, os.path.join(MODEL_DIR, "scaler_Y.save"))

# ===============================
# Torch dataset
# ===============================
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
Y_tensor = torch.tensor(Y_scaled, dtype=torch.float32)

dataset = TensorDataset(X_tensor, Y_tensor)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,     # IMPORTANT for macOS
    pin_memory=False
)

# ===============================
# Model
# ===============================
class SingleBandLSTM(nn.Module):
    def __init__(self, input_size=16, hidden_size=96, num_layers=2):
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
        # x: (B, 16)
        x = self.fc_in(x)              # (B, H)
        x = x.unsqueeze(1).repeat(1, seq_len, 1)  # (B, T, H)
        out, _ = self.lstm(x)
        return self.fc_out(out)        # (B, T, 1)

model = SingleBandLSTM(
    input_size=16,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ===============================
# Training
# ===============================
SEQ_LEN = Y_tensor.shape[1]

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for xb, yb in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()
        pred = model(xb, SEQ_LEN)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)
    print(f"Epoch {epoch+1} | MSE: {avg_loss:.6f}")

# ===============================
# Save model
# ===============================
ckpt_path = os.path.join(MODEL_DIR, "baseline_singleband.ckpt")
torch.save({
    "model_state": model.state_dict(),
    "band_index": BAND_INDEX,
    "seq_len": SEQ_LEN
}, ckpt_path)

print(f"✅ Baseline single-band LSTM saved to: {ckpt_path}")
