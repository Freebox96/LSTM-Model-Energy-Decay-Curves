import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ===============================
# Paths
# ===============================
EDC_MB_PATH = "dataset/room_acoustic_largedataset/EDC_multiband"
FEATURES_CSV = "dataset/room_acoustic_largedataset/roomFeaturesDataset.csv"

# ===============================
# Load ONE sample
# ===============================
edc_file = sorted(os.listdir(EDC_MB_PATH))[0]
edc = np.load(os.path.join(EDC_MB_PATH, edc_file))

print("Loaded multi-band EDC shape:", edc.shape)  # (T, 7)

features_df = pd.read_csv(FEATURES_CSV)
# Drop non-numeric columns (e.g., RIR ID)
numeric_features = features_df.select_dtypes(include=[np.number])

features = numeric_features.iloc[0].values.astype(np.float32)


print("Loaded feature vector shape:", features.shape)  # (16,)

# ===============================
# Convert to tensors
# ===============================
X = torch.tensor(features).unsqueeze(0).unsqueeze(1)  
# shape: (batch=1, seq=1, features=16)

Y = torch.tensor(edc).unsqueeze(0)  
# shape: (batch=1, time, bands=7)

print("X tensor shape:", X.shape)
print("Y tensor shape:", Y.shape)

# ===============================
# Define Multi-Output LSTM
# ===============================
class MultiBandLSTM(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=128, num_bands=7):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_bands)

    def forward(self, x, T):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.repeat(1, T, 1)
        out = self.fc(lstm_out)
        return out

# ===============================
# Run forward pass
# ===============================
model = MultiBandLSTM()
T = Y.shape[1]

Y_pred = model(X, T)

print("Predicted output shape:", Y_pred.shape)
