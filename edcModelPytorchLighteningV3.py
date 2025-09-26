"""
==========================================================
EDC Prediction
This Model is used for WASPAA 2024 Paper and also for ICASSP 2024 Paper (MSE Loss FUnction Only)
Last updated: 2025-09-02
==========================================================
"""

import os
import re
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import subprocess
import webbrowser
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

# This Custom Loss Functions is not used
from RT_Loss import RTLoss  # Assuming you have a custom RT_Loss module for T30/T60 calculations (This is giving issues sometimes when RT valye are not calculated properly)

# ------------------------------
# Configuration
# ------------------------------
today = datetime.today().strftime('%Y-%m-%d')
resuts_path = os.path.join('Results/ICASSP', today, time.strftime('%H-%M-%S'))
data_paths = os.path.join('Results/ICASSP')
edc_folder_path = f"dataset/room_acoustic_largedataset/EDC"
room_features_path = f"dataset/room_acoustic_largedataset/full_large_dataset.csv"
model_save_dir = f"{resuts_path}/Trained_Models"
os.makedirs(model_save_dir, exist_ok=True)

usedMethodEDC = "EDC"  # Options: "EDC", "psudoEDC"
target_length = 48000*2 # Samples per EDC
rooms_to_process = 400 # Number of Rooms to process (maximum 588 rooms available in the dataset)
absCases = 30 # Number of Absorption Cases per room (Maximum 30 cases available in the dataset)
max_files_to_load = rooms_to_process*absCases # Total files to load from the EDC folder (maximum 588*30 files available in the dataset)
batch_size = 8
input_dim = 16
customLoss = "MSE" # Options: "RTLoss", "weightedRTLoss", "MSE", "edcRIRLoss"
isScalingX = True  # Set to False if you don't want to scale the features
isScalingY = True  # Set to False if you don't want to scale the EDCs

print("Configuration: are set.")

# ------------------------------
# Load and Prepare EDC Data
# Function to extract numeric sort key from filename. Files are named like 'RIR_1_case1_edc.npy' and should be sorted by RIR number then case number
# ------------------------------
def extract_case_number(filename):
    match = re.search(f'case(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

# --- Helper function to extract RIR and case numbers ---
def extract_rir_case(fname):
    rir_match = re.search(r'RIR_(\d+)', fname)
    case_match = re.search(r'case(\d+)', fname)
    rir_num = int(rir_match.group(1)) if rir_match else float('inf')
    case_num = int(case_match.group(1)) if case_match else float('inf')
    return (rir_num, case_num)

# --- Sort files by RIR number then case number ---
edc_files = sorted([f for f in os.listdir(edc_folder_path) if f.endswith('.npy')],key=extract_rir_case)
all_edcs = []
for i, fname in enumerate(edc_files):
    if i >= max_files_to_load:
        print(f"Stopped loading after {max_files_to_load} files.")
        break
    try:
        edc = np.load(os.path.join(edc_folder_path, fname))
        edc = edc.flatten()[:target_length]
        #edc = edc.flatten()
        edc = np.pad(edc, (0, max(0, target_length - len(edc))), mode='constant')
        #edc = 10 * np.log10(edc + 1e-6)
        all_edcs.append(edc)
        print(f"Loaded {fname} with standardized shape: {edc.shape}")
    except Exception as e:
        print(f"Failed to load {fname}: {e}")

combined_edc_data = np.stack(all_edcs)[:max_files_to_load]

""" ===============================
Load room features:
16 features and remove ID column if exists
Scale if needed and reshape for LSTM input
===============================
"""

room_features_df = pd.read_csv(room_features_path).drop(columns=['ID'], errors='ignore')
room_features = room_features_df.values[:max_files_to_load]

# Scalling the Data i.e. Normalize
if isScalingX:
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(room_features)
    X_scaled = X_scaled.reshape((-1, 1, input_dim))
    joblib.dump(scaler_X, f'{resuts_path}/scaler_X_{max_files_to_load}_{target_length}.save')
else:
    X_scaled = room_features
    X_scaled = X_scaled.reshape((-1, 1, input_dim))

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

#X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

train_dataset = EDCDataset(X_train, y_train)
val_dataset = EDCDataset(X_val, y_val)
test_dataset = EDCDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

train_loss_list = []
val_loss_list = []

# ------------------------------
# Lightning Module
# ------------------------------
class EDCModel(pl.LightningModule):
    def __init__(self, input_dim, target_length, rt_weights=(2.0, 1.0, 0.5)):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(input_dim, 128, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 2048)
        self.fc2 = nn.Linear(2048, target_length)
        
        self.val_losses = []
        self.train_losses = []
                        
        if customLoss == "RTLoss":
            self.criterion = RTLoss()
        elif customLoss == "weightedRTLoss":
            #self.criterion = WeightedRTLoss(rt_weights=(0.5, 1.0, 2.0))  # Prefer RT60 more heavily
            #self.criterion = WeightedRTLoss(rt_weights=(0.5, 1.0, 0.5))  # Prefer RT30 more heavily
            self.criterion = WeightedRTLoss(rt_weights=rt_weights)  # Prefer RT20 more heavily
        else:
            self.criterion = nn.MSELoss()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        x = torch.relu(self.fc1(h_n[-1]))
        x = self.dropout(x)
        return self.fc2(x)

    def training_step(self, batch, batch_idx):
        
        X, y = batch
        y_hat = self(X)
        if customLoss == "RTLoss":
            loss = self.criterion(y_hat, y)
        elif customLoss == "weightedRTLoss":
            loss = self.criterion(y_hat, y)
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
        if customLoss == "RTLoss":
            val_loss = self.criterion(y_hat, y)
        elif customLoss == "weightedRTLoss":
            val_loss = self.criterion(y_hat, y)
        else:
            val_loss = self.criterion(y_hat, y)
        val_loss = self.criterion(y_hat, y)
        val_mae = torch.mean(torch.abs(y_hat - y))
        self.val_losses.append(val_loss.item())
        
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mae', val_mae, on_step=True, on_epoch=True, prog_bar=True, logger=True)
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
if customLoss == "RTLoss":
    model = EDCModel(input_dim=input_dim, target_length=target_length, rt_weights=(2.0, 1.0, 0.5))
else:
    model = EDCModel(input_dim=input_dim, target_length=target_length)
    # --- Uncomment the following line to use the Transformer model instead ---
    #model = EDCTransformerModel(input_dim=input_dim, target_length=target_length)


early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")
checkpoint = ModelCheckpoint(monitor="val_loss", dirpath=model_save_dir, filename="best_model", save_top_k=1)
CHECKPOINT_PATH = os.path.join(model_save_dir, "best_model.ckpt")

maxEpochs = 200
resume_training = False

if os.path.exists(CHECKPOINT_PATH):
    print(f"Found checkpoint at {CHECKPOINT_PATH}. Resuming training from best model.")
    model = EDCModel.load_from_checkpoint(
        CHECKPOINT_PATH, 
        input_dim=input_dim, 
        target_length=target_length
    )
    resume_training = True

trainer = pl.Trainer(
    max_epochs=maxEpochs,
    callbacks=[early_stop, checkpoint],
    accelerator='auto',
    devices='auto',
    log_every_n_steps=5,
    enable_progress_bar=True,
    enable_model_summary=True,
    resume_from_checkpoint=CHECKPOINT_PATH if resume_training else None
)

trainer.fit(model, train_loader, val_loader)
print(f"Model saved at: {checkpoint.best_model_path}")
print("...")


""" # Old Model With out Resume Training Code
trainer = pl.Trainer(
    max_epochs=maxEpochs,
    callbacks=[early_stop, checkpoint],
    accelerator='auto',
    devices='auto',
    log_every_n_steps=5,
    enable_progress_bar=True,
    enable_model_summary=True,    
)

trainer.fit(model, train_loader, val_loader)
print(f"Model saved at: {checkpoint.best_model_path}")
"""

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
print("...")

# ------------------------------
# Save Outputs
# ------------------------------
isSave = "npy"
if isSave == "npy":
    np.save(f"{resuts_path}/predicted_edcs_sample_{max_files_to_load}_{target_length}.npy", preds)
    np.save(f"{resuts_path}/actual_edcs_sample_{max_files_to_load}_{target_length}.npy", targets)
    np.save(f"{resuts_path}/predicted_edcs_sample_rescaled_{max_files_to_load}_{target_length}.npy", preds_rescaled)
    np.save(f"{resuts_path}/actual_edcs_sample_rescaled_{max_files_to_load}_{target_length}.npy", targets_rescaled)
else:
    np.savetxt(f"{resuts_path}/predicted_edcs_sample_{max_files_to_load}_{target_length}.csv", preds, delimiter=",")
    np.savetxt(f"{resuts_path}/actual_edcs_sample_{max_files_to_load}_{target_length}.csv", targets, delimiter=",")
    np.savetxt(f"{resuts_path}/predicted_edcs_sample_rescaled_{max_files_to_load}_{target_length}.csv", preds_rescaled, delimiter=",")
    np.savetxt(f"{resuts_path}/actual_edcs_sample_rescaled_{max_files_to_load}_{target_length}.csv", targets_rescaled, delimiter=",")

print(f"Saved predicted EDCs to {resuts_path}")

# ------------------------------

# Save metadata
metadata = {
    "target_length": target_length,
    "rooms_to_process": max_files_to_load,
    "batch_size": batch_size,
    "input_dim": input_dim,
    "customLoss": customLoss,
    "maxEpochs": maxEpochs,
    "mae": float(np.mean(mae)),  # or mae.tolist() if it's an array
    "mse": float(np.mean(mse)),  # or mse.tolist()
    "model_save_path": str(checkpoint.best_model_path),
    "scaler_X": bool(isScalingX),
    "scaler_y": bool(isScalingY),
    "method": usedMethodEDC,
    "Training Datraset Size": X_train.shape[0],
    "Validation Dataset Size": X_val.shape[0],
    "Test Dataset Size": X_test.shape[0]    
}

# Save metadata to a JSON file

json_path = os.path.join(resuts_path, "experiment_metadata.json")
with open(json_path, "w") as f:
    json.dump(metadata, f, indent=4)
print(f"Metadata saved to {json_path}")

print("Training and evaluation completed successfully!")