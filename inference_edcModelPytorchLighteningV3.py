import os
import torch
import joblib
import numpy as np
import pandas as pd
import torch.nn as nn
import soundfile as sf
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from sklearn.metrics import mean_squared_error

# -----------------------------
# Define the model (same as training)


class EDCModel(pl.LightningModule):
    def __init__(self, input_dim, target_length):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 128, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 2048)
        self.fc2 = nn.Linear(2048, target_length)

    def forward(self, x):
        x = x.to(self.device)
        _, (h_n, _) = self.lstm(x)
        x = torch.relu(self.fc1(h_n[-1]))
        x = self.dropout(x)
        return self.fc2(x)

# -----------------------------
# Helper: Random Sign-Sticky RIR Reconstruction

def reconstruct_random_sign_sticky(edc, stickiness=0.90):
    diff_edc = -np.diff(edc, append=0)
    diff_edc = np.clip(diff_edc, 0, None)
    rir_mag = np.sqrt(diff_edc)
    signs = np.empty_like(rir_mag)
    last_sign = 1
    for i, mag in enumerate(rir_mag):
        if mag == 0:
            signs[i] = last_sign
        else:
            signs[i] = last_sign if np.random.rand() < stickiness else -last_sign
            last_sign = signs[i]
    return rir_mag * signs

# -----------------------------
# Inference function

def infer_and_analyze(
    checkpoint_path,
    scaler_X_path,
    scaler_y_path,
    new_features,
    actual_edc,
    input_dim=16,
    target_length=96000,
    fs=48000,
    save_dir="inference_results"
):
    os.makedirs(save_dir, exist_ok=True)
    epsilon = 1e-18
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load scalers
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)

    # Scale input features
    new_features_scaled = scaler_X.transform(new_features.reshape(1, -1))
    new_features_scaled = new_features_scaled.reshape(1, 1, input_dim)

    # Load model
    model = EDCModel(input_dim=input_dim, target_length=target_length)
    model = EDCModel.load_from_checkpoint(checkpoint_path, input_dim=input_dim, target_length=target_length)
    model.to(DEVICE)
    model.eval()

    # Predict
    with torch.no_grad():
        X_tensor = torch.tensor(new_features_scaled, dtype=torch.float32).to(DEVICE)  # Move input to device
        pred_scaled = model(X_tensor).cpu().numpy() 

    # Inverse transform prediction
    pred_edc = scaler_y.inverse_transform(pred_scaled)[0]

    # Align actual EDC
    actual_edc = actual_edc[:target_length]

    # Compute MSE
    mse = mean_squared_error(actual_edc, pred_edc)
    print(f"EDC MSE = {mse:.6f}")

    # Reconstruct RIRs
    pred_rir = reconstruct_random_sign_sticky(pred_edc)
    actual_rir = reconstruct_random_sign_sticky(actual_edc)
    
    actual_fft = np.abs(np.fft.rfft(actual_rir))
    pred_fft = np.abs(np.fft.rfft(pred_rir))
    freqs = np.fft.rfftfreq(len(actual_rir), d=1/fs)

    # Normalize FFT to maximum of actual RIR (0 dB reference)
    actual_fft_db = 20 * np.log10(actual_fft / actual_fft.max() + epsilon)
    pred_fft_db   = 20 * np.log10(pred_fft / actual_fft.max() + epsilon)  # keep same reference

    # Save RIRs
    #sf.write(os.path.join(save_dir, "predicted_rir.wav"), pred_rir, fs)
    #sf.write(os.path.join(save_dir, "actual_rir.wav"), actual_rir, fs)

    # Compute FFTs
    pred_fft = np.abs(np.fft.rfft(pred_rir))
    actual_fft = np.abs(np.fft.rfft(actual_rir))
    freqs = np.fft.rfftfreq(len(pred_rir), 1/fs)

    # ===========================================================================
    # Plot results
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # Subplot 1: EDCs
    axs[0].plot(10 * np.log10(actual_edc + epsilon), label="Actual EDC")
    axs[0].plot(10 * np.log10(pred_edc + epsilon), label="Predicted EDC")
    axs[0].set_title("EDCs (dB)")
    axs[0].set_xlabel("Samples")
    axs[0].set_ylabel("Energy (dB)")
    axs[0].legend()
    axs[0].grid(True)

    # Subplot 2: RIRs
    axs[1].plot(actual_rir, label="Actual RIR")
    axs[1].plot(pred_rir, label="Predicted RIR")
    axs[1].set_title("Reconstructed RIRs")
    axs[1].set_xlabel("Samples")
    axs[1].set_ylabel("Amplitude")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(freqs, actual_fft_db, label="Actual FFT")
    axs[2].plot(freqs, pred_fft_db, label="Predicted FFT")
    axs[2].set_title("FFT Magnitude Spectrum (0 dB = Max of Actual RIR)")
    axs[2].set_xlabel("Frequency (Hz)")
    axs[2].set_xscale("log")
    axs[2].set_ylabel("Magnitude (dB)")
    axs[2].legend()
    axs[2].grid(True)
    
    fig.tight_layout()
    plot_path = os.path.join(save_dir, "comparison_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.show()
    
    # ===========================================================================
    # Plot results - Zoomed in Version
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # Subplot 1: EDCs
    axs[0].plot(10 * np.log10(actual_edc + epsilon), label="Actual EDC")
    axs[0].plot(10 * np.log10(pred_edc + epsilon), label="Predicted EDC")
    axs[0].set_title("EDCs (dB)")
    axs[0].set_xlabel("Samples")
    axs[0].set_xlim(0, 2000)
    axs[0].set_ylim(-10,5)
    axs[0].set_ylabel("Energy (dB)")
    axs[0].legend()
    axs[0].grid(True)

    # Subplot 2: RIRs
    axs[1].plot(abs(actual_rir), label="Actual RIR")
    axs[1].plot(abs(pred_rir), linestyle="--", label="Predicted RIR")
    axs[1].set_title("Reconstructed RIRs ABS only for comparison")
    axs[1].set_xlabel("Samples")
    axs[1].set_xlim(0, 2000)
    axs[1].set_ylabel("Amplitude")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(freqs, actual_fft_db, label="Actual FFT")
    axs[2].plot(freqs, pred_fft_db, label="Predicted FFT")
    axs[2].set_title("FFT Magnitude Spectrum (0 dB = Max of Actual RIR)")
    axs[2].set_xlabel("Frequency (Hz)")
    axs[2].set_xscale("log")
    axs[2].set_ylabel("Magnitude (dB)")
    axs[2].legend()
    axs[2].grid(True)
    
    fig.tight_layout()
    plot_path = os.path.join(save_dir, "comparison_plot_Zoom.png")
    plt.savefig(plot_path, dpi=300)
    plt.show()
    
    print(f"Results saved in: {save_dir}")
    return mse, pred_edc, actual_edc, pred_rir, actual_rir


if __name__ == "__main__":
    
    # Configuration used during training, keep consistent (No changes here)
    FS = 48000
    target_length = FS*2  # Samples per EDC
    rooms_to_process = 200
    absCases = 30
    max_files_to_load = rooms_to_process*absCases
    batch_size = 8
    input_dim = 16
    # ======================================================
    # Paths (change as needed)
    room_features_csv = f"dataset/room_acoustic_largedataset/full_large_dataset.csv"
    edc_folder = f"dataset/room_acoustic_largedataset/EDC"
    checkpoint_path = f"Results 2025-09-13/22-16-21/Trained_Models/best_model.ckpt"
    scaler_X_path = f"Results 2025-09-13/22-16-21/scaler_X_{max_files_to_load}_{target_length}.save"
    scaler_y_path = f"Results 2025-09-13/22-16-21/scaler_edc_{max_files_to_load}_{target_length}.save"
    # ======================================================
    # Load room features CSV and select a random row
    df_features = pd.read_csv(room_features_csv)
    # Remove ID column from features (keep for mapping)
    room_ids = df_features.iloc[:, 0].values  # first column: ID
    features_only = df_features.drop(columns=[df_features.columns[0]]).values
    # Select a random row
    rng = np.random.default_rng()
    #rand_idx = rng.integers(6000, len(features_only))
    rand_idx = rng.integers(1, 6000)
    selected_features = features_only[rand_idx]
    selected_room_id = room_ids[rand_idx]
    print(f"Selected room ID: {selected_room_id}")
    print(f"Selected features shape: {selected_features.shape}")

    # Load corresponding EDC
    # or adapt pattern to your file naming
    edc_filename = f"{selected_room_id}.npy"
    edc_path = os.path.join(edc_folder, edc_filename)

    if not os.path.exists(edc_path):
        raise FileNotFoundError(f"EDC file not found: {edc_path}")

    actual_edc = np.load(edc_path).flatten()
    # Pad or truncate to target length
    actual_edc = np.pad(actual_edc, (0, max(
        0, target_length - len(actual_edc))), mode='constant')[:target_length]
    print(f"Loaded EDC shape: {actual_edc.shape}")

    # Run inference and analysis
    infer_and_analyze(
        checkpoint_path,
        scaler_X_path,
        scaler_y_path,
        selected_features,
        actual_edc,
        input_dim=input_dim,
        target_length=target_length,
        fs=FS,
        save_dir = f"inference_results"
    )
    
    print("Inference and analysis complete.")
    
