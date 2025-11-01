"""
==========================================================
EDC Prediction with Custom EDC+RIR Loss
Last updated: 2025-09-14
==========================================================
This script trains an LSTM model to predict Energy Decay Curves (EDCs) from room features.
It includes a custom loss function that combines EDC and RIR reconstruction losses.
The model is implemented using PyTorch Lightning for streamlined training and evaluation.
Key Features:
- Custom Loss Function: Combines EDC and RIR losses with adjustable weights.
- Data Handling: Loads and preprocesses EDC data and room features.
- Model Architecture: LSTM-based model for sequence prediction.
- Training: Includes early stopping and model checkpointing.
- Evaluation: Computes MAE and MSE, saves predictions and metadata.
- Visualization: Plots training and validation loss over time.
==========================================================
Cite:
If you use this code, please cite the following paper:
@inproceedings{tui2026edc,
    title={ROOM IMPULSE RESPONSE PREDICTION WITH NEURAL NETWORKS: FROM ENERGY DECAY CURVES TO PERCEPTUAL VALIDATION},
    author={Imran Muhammad, Gerald Schuller},
    booktitle={2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    year={2026},
    organization={IEEE}
    }
"""

import os
import re
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# ------------------------------
# Configuration
# ------------------------------
today = datetime.today().strftime('%Y-%m-%d')
resuts_path = os.path.join('Results/', today, time.strftime('%H-%M-%S'))
data_paths = os.path.join('Results/')
edc_folder_path = f"dataset/room_acoustic_largedataset/EDC"
room_features_path = f"dataset/room_acoustic_largedataset/roomFeaturesDataset.csv"
model_save_dir = f"{resuts_path}/Trained_Models"
os.makedirs(model_save_dir, exist_ok=True)

usedMethodEDC = "EDC"  # Options: "EDC", "psudoEDC"
target_length = 48000*2 # Samples per EDC
rooms_to_process = 200 # Number of rooms to process
absCases = 30 # Number of absorption cases per room
max_files_to_load = rooms_to_process*absCases # Total files to load from EDC folder
batch_size = 8 # Batch size for training
input_dim = 16 # Number of input features (room features)
customLoss = "MSE" # Options: "MSE", "edcRIRLoss"
isScalingX = True
isScalingY = True

# ------------------------------
# Load and Prepare EDC Data
# ------------------------------
def extract_rir_case(fname):
    rir_match = re.search(r'RIR_(\d+)', fname)
    case_match = re.search(r'case(\d+)', fname)
    rir_num = int(rir_match.group(1)) if rir_match else float('inf')
    case_num = int(case_match.group(1)) if case_match else float('inf')
    return (rir_num, case_num)

edc_files = sorted([f for f in os.listdir(edc_folder_path) if f.endswith('.npy')], key=extract_rir_case)
all_edcs = []
for i, fname in enumerate(edc_files):
    if i >= max_files_to_load:
        print(f"Stopped loading after {max_files_to_load} files.")
        break
    try:
        edc = np.load(os.path.join(edc_folder_path, fname))
        edc = edc.flatten()[:target_length]
        edc = np.pad(edc, (0, max(0, target_length - len(edc))), mode='constant')
        all_edcs.append(edc)
        print(f"Loaded {fname} with standardized shape: {edc.shape}")
    except Exception as e:
        print(f"Failed to load {fname}: {e}")

combined_edc_data = np.stack(all_edcs)[:max_files_to_load]

# ------------------------------
# Load room features
# ------------------------------
room_features_df = pd.read_csv(room_features_path).drop(columns=['ID'], errors='ignore')
room_features = room_features_df.values[:max_files_to_load]

if isScalingX:
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(room_features)
    X_scaled = X_scaled.reshape((-1, 1, input_dim))
    joblib.dump(scaler_X, f'{resuts_path}/scaler_X_{max_files_to_load}_{target_length}.save')
else:
    X_scaled = room_features.reshape((-1, 1, input_dim))

if isScalingY:
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(combined_edc_data)
    joblib.dump(scaler_y, f'{resuts_path}/scaler_edc_{max_files_to_load}_{target_length}.save')
else:
    y_scaled = combined_edc_data

# ------------------------------
# PyTorch Dataset
# ------------------------------
class EDCDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

train_dataset = EDCDataset(X_train, y_train)
val_dataset = EDCDataset(X_val, y_val)
test_dataset = EDCDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

train_loss_list = []
val_loss_list = []

# ------------------------------
# Custom Loss Function for EDC + RIR
# ------------------------------
class EDCRIRLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(EDCRIRLoss, self).__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, return_components: bool = False):
        edc_loss = F.mse_loss(y_pred, y_true, reduction='mean')
        rir_pred = y_pred[:, 1:] - y_pred[:, :-1]
        rir_true = y_true[:, 1:] - y_true[:, :-1]
        rir_loss = F.mse_loss(rir_pred, rir_true, reduction='mean')
        total = self.alpha * edc_loss + self.beta * rir_loss
        if return_components:
            return total, edc_loss, rir_loss
        return total

# ------------------------------
# Lightning Module
# ------------------------------
class EDCModel(pl.LightningModule):
    def __init__(self, input_dim, target_length, rt_weights=(2.0,1.0,0.5), edc_rir_alpha=1.0, edc_rir_beta=0.5):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(input_dim, 128, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 2048)
        self.fc2 = nn.Linear(2048, target_length)

        if customLoss == "RTLoss":
            self.criterion = RTLoss()
        elif customLoss == "weightedRTLoss":
            self.criterion = WeightedRTLoss(rt_weights=rt_weights)
        elif customLoss == "edcRIRLoss":
            self.criterion = EDCRIRLoss(alpha=edc_rir_alpha, beta=edc_rir_beta)
        else:
            self.criterion = nn.MSELoss()

        self.val_losses = []
        self.train_losses = []

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        x = torch.relu(self.fc1(h_n[-1]))
        x = self.dropout(x)
        return self.fc2(x)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        if isinstance(self.criterion, EDCRIRLoss):
            total_loss, edc_term, rir_term = self.criterion(y_hat, y, return_components=True)
            loss = total_loss
            self.log('train_edc_loss', edc_term, on_step=True, on_epoch=True, prog_bar=False)
            self.log('train_rir_loss', rir_term, on_step=True, on_epoch=True, prog_bar=False)
        else:
            loss = self.criterion(y_hat, y)
        mae = torch.mean(torch.abs(y_hat - y))
        self.train_losses.append(loss.item())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_mae', mae, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        global train_loss_list
        avg_train_loss = sum(self.train_losses) / len(self.train_losses)
        train_loss_list.append(avg_train_loss)
        self.log('train_loss_epoch', avg_train_loss, prog_bar=True)
        self.train_losses.clear()

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        if isinstance(self.criterion, EDCRIRLoss):
            total_loss, edc_term, rir_term = self.criterion(y_hat, y, return_components=True)
            val_loss = total_loss
            self.log('val_edc_loss', edc_term, on_step=False, on_epoch=True, prog_bar=False)
            self.log('val_rir_loss', rir_term, on_step=False, on_epoch=True, prog_bar=False)
        else:
            val_loss = self.criterion(y_hat, y)
        val_mae = torch.mean(torch.abs(y_hat - y))
        self.val_losses.append(val_loss.item())
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mae', val_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def on_validation_epoch_end(self):
        global val_loss_list
        avg_val_loss = sum(self.val_losses) / len(self.val_losses)
        val_loss_list.append(avg_val_loss)
        self.log('val_loss_epoch', avg_val_loss, prog_bar=True)
        self.val_losses.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# ------------------------------
# Logging and Training
# ------------------------------
log_dir = f"{resuts_path}/lightning_logs"
logger = TensorBoardLogger(save_dir=log_dir, name=f"edc_model_{max_files_to_load}_{target_length}")

model = EDCModel(input_dim=input_dim, target_length=target_length, edc_rir_alpha=1.0, edc_rir_beta=0.5)

early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")
checkpoint = ModelCheckpoint(monitor="val_loss", dirpath=model_save_dir, filename="best_model", save_top_k=1)

maxEpochs = 200
trainer = pl.Trainer(
    max_epochs=maxEpochs,
    callbacks=[early_stop, checkpoint],
    accelerator='auto',
    devices='auto',
    log_every_n_steps=5,
    enable_progress_bar=True,
    enable_model_summary=True,
    gradient_clip_val=1.0,
)

trainer.fit(model, train_loader, val_loader)
print(f"Model saved at: {checkpoint.best_model_path}")

# Plot Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Validation Loss')
plt.xlabel('Batch (not Epoch)')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Progress')
plt.legend()
plt.grid(True)
plt.savefig(f"{resuts_path}/loss_plot.png")
plt.show()

# ------------------------------
# Evaluate
# ------------------------------
model.eval()
preds, targets = [], []
for X, y in val_loader:
    with torch.no_grad():
        output = model(X)
    preds.append(output.numpy())
    targets.append(y.numpy())

preds = np.vstack(preds)
targets = np.vstack(targets)

if isScalingY:
    preds_rescaled = scaler_y.inverse_transform(preds)
    targets_rescaled = scaler_y.inverse_transform(targets)
else:
    preds_rescaled = preds
    targets_rescaled = targets

mae = mean_absolute_error(targets_rescaled, preds_rescaled)
mse = mean_squared_error(targets_rescaled, preds_rescaled)
print(f"Test MAE: {mae:.5f}, MSE: {mse:.5f}")

# Save outputs
np.save(f"{resuts_path}/predicted_edcs_sample_{max_files_to_load}_{target_length}.npy", preds)
np.save(f"{resuts_path}/actual_edcs_sample_{max_files_to_load}_{target_length}.npy", targets)
np.save(f"{resuts_path}/predicted_edcs_sample_rescaled_{max_files_to_load}_{target_length}.npy", preds_rescaled)
np.save(f"{resuts_path}/actual_edcs_sample_rescaled_{max_files_to_load}_{target_length}.npy", targets_rescaled)

# Metadata
metadata = {
    "target_length": target_length,
    "rooms_to_process": max_files_to_load,
    "batch_size": batch_size,
    "input_dim": input_dim,
    "customLoss": customLoss,
    "maxEpochs": maxEpochs,
    "mae": float(np.mean(mae)),
    "mse": float(np.mean(mse)),
    "model_save_path": str(checkpoint.best_model_path),
    "scaler_X": bool(isScalingX),
    "scaler_y": bool(isScalingY),
    "method": usedMethodEDC,
    "Training Dataset Size": X_train.shape[0],
    "Validation Dataset Size": X_val.shape[0],
    "Test Dataset Size": X_test.shape[0]
}

json_path = os.path.join(resuts_path, "experiment_metadata.json")
with open(json_path, "w") as f:
    json.dump(metadata, f, indent=4)
print(f"Metadata saved to {json_path}")

print("Training and evaluation completed successfully!")
