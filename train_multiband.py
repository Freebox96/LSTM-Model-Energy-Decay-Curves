# train_multiband.py
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

EPOCHS = 20
BATCH_SIZE = 16
LR = 1e-3
DEVICE = "cpu"
N_BANDS = 7

os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================
# Multiband LSTM Model (SAFE TO IMPORT)
# ===============================
class MultiBandLSTM(nn.Module):
    def __init__(self, input_size=16, hidden_size=128, num_layers=2, n_bands=7, seq_len=1440):
        super().__init__()
        self.seq_len = seq_len
        self.fc_in = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_size, n_bands)

    def forward(self, x):
        # x: (B, 16)
        x = self.fc_in(x)                       # (B, H)
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)  # (B, T, H)
        out, _ = self.lstm(x)                   # (B, T, H)
        return self.fc_out(out)                 # (B, T, 7)

# ===============================
# Training function (NOT auto-run)
# ===============================
def train():
    # Load dataset
    X = np.load(os.path.join(DATASET_DIR, "X_multiband.npy"))
    Y = np.load(os.path.join(DATASET_DIR, "Y_multiband_downsampled.npy"))

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    # Safety
    X[~np.isfinite(X)] = 0.0
    Y[~np.isfinite(Y)] = 0.0

    # Scaling
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)

    N, T, B = Y.shape
    Y_scaled = scaler_Y.fit_transform(
        Y.reshape(N * T, B)
    ).reshape(N, T, B)

    joblib.dump(scaler_X, os.path.join(MODEL_DIR, "scaler_X_multiband.save"))
    joblib.dump(scaler_Y, os.path.join(MODEL_DIR, "scaler_Y_multiband.save"))

    # Torch
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_scaled, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_tensor, Y_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Model
    model = MultiBandLSTM(seq_len=T).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for xb, yb in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1} | MSE: {epoch_loss / len(loader):.6f}")

    # Save
    ckpt_path = os.path.join(MODEL_DIR, "multiband_lstm.ckpt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "seq_len": T,
            "n_bands": N_BANDS
        },
        ckpt_path
    )

    print(f"✅ Multiband LSTM saved to: {ckpt_path}")

# ===============================
# ENTRY POINT (THIS IS THE KEY)
# ===============================
if __name__ == "__main__":
    train()
