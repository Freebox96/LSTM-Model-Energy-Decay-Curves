import torch
from baseline_singleband_lstm import SingleBandLSTM
import joblib
import numpy as np

ckpt = torch.load("Models/baseline_singleband.ckpt")
model = SingleBandLSTM(input_size=16)
model.load_state_dict(ckpt["model_state"])
model.eval()

# Load scalers
scaler_X = joblib.load("Models/scaler_X.save")
scaler_Y = joblib.load("Models/scaler_Y.save")

# Predict on new X
X_new = np.load("dataset/room_acoustic_largedataset/final_dataset/X_multiband.npy")
X_new_scaled = scaler_X.transform(X_new)
import torch
y_pred = model(torch.tensor(X_new_scaled, dtype=torch.float32))
y_pred = y_pred.detach().numpy()
# Inverse scale
y_pred_inv = scaler_Y.inverse_transform(y_pred.reshape(-1, 7)).reshape(y_pred.shape)
