"""
================================================
Energy Decay Curve (EDC) Prediction Using LSTM

Author: Muhammad Imran
Date: 2025-05-02
================================================
- Step 1: Load data
- Step 2: Normalize features and reshape
- Step 3: Define and train LSTM model
- Step 4: Evaluate and visualize results
- Step 5: Compute and compare RT values (T30/T60)
- Step 6: Save results to CSV
- Step 7: Plot results
================================================
"""

# Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import helperFunctions.aacousticsParameters as ap

# Config and Paths
edc_folder_path = r"dataset/room_acoustic_parameters/EDC"
room_features_path = r"dataset/mim_full_dataset.csv"
target_length = 44100  # Samples per EDC
rooms_to_process = 588  # Number of Rooms to process
model_save_path = f"lstm_edc_model_realDataset_{rooms_to_process}0_{target_length}.h5"

"""
Step 1: Load Room Features and EDC Data
"""
npy_files = [f for f in os.listdir(edc_folder_path) if f.endswith('.npy')]
all_edcs = []

# Load EDC files and standardize their lengths
for fname in npy_files:
    try:
        edc_array = np.load(os.path.join(edc_folder_path, fname))
        edc_array = edc_array.flatten() if edc_array.ndim > 1 else edc_array
        edc_array = np.pad(edc_array, (0, max(0, target_length - len(edc_array))), mode='constant')[:target_length]
        all_edcs.append(edc_array)
        print(f"Loaded {fname} with standardized shape: {edc_array.shape}")
        #time.sleep(0.5)
    except Exception as e:
        print(f"Failed to load {fname}: {e}")
combined_edc_data = np.stack(all_edcs, axis=0)

# Select only the first "rooms_to_process" data for processing
combined_edc_data = combined_edc_data[:rooms_to_process*10]
print(f"\nFinal combined EDC shape: {combined_edc_data.shape}")

# Load room features
room_features_df = pd.read_csv(room_features_path).drop(columns=['ID'], errors='ignore')
room_features = room_features_df.values

# Select only the first "rooms_to_process" data for processing
room_features = room_features[:rooms_to_process*10]

# Print the number of rooms and time steps Selected
num_rooms = room_features.shape[0]
num_time_steps = combined_edc_data.shape[1]
print(f"Loaded {num_rooms} rooms and {num_time_steps} EDC time steps.")

# Check few sample EDCs to visualize
checkSampleECD = True
logScale = True

if checkSampleECD:
    plt.figure(figsize=(12, 6))
    num_plots = 12
    if logScale == False:
        for i in range(num_plots):
            plt.subplot(3, 4, i + 1)
            plt.plot(combined_edc_data[i])
            plt.title(f'EDC Sample {i + 1}')
            plt.xlabel('Time (samples)')
            plt.ylabel('Energy')
            plt.grid(True)
        plt.suptitle('Sample Energy Decay Curves (EDCs)', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    else:
        for i in range(num_plots):
            plt.subplot(3, 4, i + 1)
            edc_db = 10 * np.log10(combined_edc_data[i] / np.max(combined_edc_data[i]) + 1e-10)
            plt.plot(edc_db)
            plt.title(f'EDC Sample {i + 1} (dB)')
            plt.xlabel('Time (samples)')
            plt.ylabel('Energy (dB)')
            plt.grid(True)
    plt.suptitle('Sample EDCs in Decibels', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    print("Sample EDCs plotted.")

"""
Step 2: Normalize and Prepare Data for LSTM
"""
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(room_features)
X_lstm_ready = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
joblib.dump(scaler_X, f'scaler_X_{rooms_to_process}0_{target_length}.save')

scaler_edc = MinMaxScaler()
y_scaled = scaler_edc.fit_transform(combined_edc_data)
joblib.dump(scaler_edc, f'scaler_edc_{rooms_to_process}0_{target_length}.save')

X_train, X_test, y_train, y_test = train_test_split(X_lstm_ready, y_scaled, test_size=0.2, random_state=42)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test : {X_test.shape}, y_test : {y_test.shape}")

np.save(f'X_train_{rooms_to_process}0_{target_length}.npy', X_train)
np.save(f'X_test_{rooms_to_process}0_{target_length}.npy', X_test)
np.save(f'y_train_{rooms_to_process}0_{target_length}.npy', y_train)
np.save(f'y_test_{rooms_to_process}0_{target_length}.npy', y_test)
print("Data prepared and saved.")

"""	
Step 3: Define Custom Loss Functions (Optional)
"""
# Custom loss functions can be defined here if needed.
def edc_loss(y_true, y_pred):
    epsilon = 1e-8
    y_true_clipped = tf.clip_by_value(y_true, epsilon, tf.reduce_max(y_true))
    y_pred_clipped = tf.clip_by_value(y_pred, epsilon, tf.reduce_max(y_pred))
    log_y_true = tf.math.log(y_true_clipped)
    log_y_pred = tf.math.log(y_pred_clipped)
    weights = tf.linspace(1.0, 0.1, tf.shape(y_pred)[-1])
    return tf.reduce_mean(weights * tf.square(log_y_pred - log_y_true))

def log_cosh_loss(y_true, y_pred):
    error = y_pred - y_true
    loss = error + tf.math.softplus(-2. * error) - tf.math.log(2.0)
    return tf.reduce_mean(loss)

"""
Step 4: Build and Train LSTM Model
"""
model = Sequential([
    LSTM(128, input_shape=(1, 16), return_sequences=False),
    Dropout(0.3),
    Dense(2048, activation='relu'),
    Dropout(0.3),
    Dense(target_length, activation='linear')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mean_squared_error',
              metrics=['mae'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=8,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop],
                    verbose=1)

# Save the model
model.save(model_save_path)
print(f"Model saved at: {model_save_path}")

# Plot Losses
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""
Step 5: Evaluate Model
"""
test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {test_loss}")

# Rescale predictions and actual values
predicted_edc = model.predict(X_test)
predicted_edc_rescaled = scaler_edc.inverse_transform(predicted_edc)
actual_edc_rescaled = scaler_edc.inverse_transform(y_test)

# Print prediction accuracy
mae = mean_absolute_error(actual_edc_rescaled, predicted_edc_rescaled)
mse = mean_squared_error(actual_edc_rescaled, predicted_edc_rescaled)
print(f"\nPrediction Accuracy:\nMAE: {mae:.5f}\nMSE: {mse:.5f}")

"""
Step 6: Plot Predicted vs Actual EDCs
"""
# Plotting the first 9 predicted vs actual EDCs in actual scale
for n in range(9):
    plt.subplot(3, 3, n+1)
    plt.plot(predicted_edc_rescaled[n+1], label='Predicted')
    plt.plot(actual_edc_rescaled[n+1], label='Actual', linestyle='dashed')
    plt.title(f'Room {n + 1}')
    plt.xlabel('Time (samples)')
    plt.ylabel('Energy Decay')
    plt.legend(fontsize='x-small')
plt.tight_layout()
plt.show()

# Save Scaled Predicted and Actual EDC Samples to CSV
np.savetxt(f"predicted_edcs_sample_{rooms_to_process}0_{target_length}.csv", predicted_edc, delimiter=",")
np.savetxt(f"actual_edcs_sample_{rooms_to_process}0_{target_length}.csv", y_test, delimiter=",")

# Save Rescaled Predicted and Actual EDC Samples to CSV
np.savetxt(f"predicted_edcs_sample_rescaled_{rooms_to_process}0_{target_length}.csv", predicted_edc_rescaled, delimiter=",")
np.savetxt(f"actual_edcs_sample_rescaled_{rooms_to_process}0_{target_length}.csv", actual_edc_rescaled, delimiter=",")
# === End of Script ===
