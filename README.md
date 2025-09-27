# Room Impulse Response (RIR) Reconstruction using Deep Learning

## 📖 Introduction

This repository contains a deep learning framework for **reconstructing Room Impulse Responses (RIRs)** from Energy Decay Curves (EDCs) using a **PyTorch Lightning LSTM model**.

The workflow is as follows:

1. Input **room geometry**, **absorption coefficients**, and **source/receiver positions**.
2. The model predicts the **EDC** of the room.
3. The predicted EDC is converted back into a **time-domain RIR** using stochastic reconstruction (Random Sign / Random Sign Sticky).
4. The framework outputs:

   * **Predicted EDC vs. Ground Truth**
   * **Reconstructed RIRs**
   * **Frequency-domain magnitude response**
   * Error metrics such as **Mean Squared Error (MSE)**

This approach is efficient for **early-stage architectural design**, **auralization**, and **real-time acoustic parameter estimation**.

---

## ⚙️ Installation

Clone this repository:

```bash
git clone https://github.com/your-username/RIR-Reconstruction.git
cd RIR-Reconstruction
```

### Option 1 – Using pip

```bash
pip install -r requirements.txt
```

### Option 2 – Using conda

```bash
conda env create -f environment.yml
conda activate rir-reconstruction
```

---

## ▶️ Running Inference

The main script is **`inference_edcModelPytorchLighteningV3.py`**.

### Step 1 – Prepare Files

Make sure you have:

* A **trained model checkpoint** (`.ckpt`)
* The **scalers** (`scaler_X_*.save` and `scaler_edc_*.save`)
* The **room features CSV** (`full_large_dataset.csv`)
* The **ground truth EDCs** (in `.npy` format inside the EDC folder)

### Step 2 – Run Inference

```bash
python inference_edcModelPytorchLighteningV3.py
```

By default, the script:

* Selects a **random room configuration** from the dataset.
* Loads the **trained model** and **scalers**.
* Predicts the EDC.
* Reconstructs the RIR using the **Random Sign Sticky method**.
* Saves plots, `.wav` files, and results inside `Results/.../inference_results`.

---

## 🏋️ Training the Model

If you want to train the LSTM model from scratch, follow these steps:

### Step 1 – Dataset Preparation

* Place your **room features CSV** (`full_large_dataset.csv`) inside:

  ```
  dataset/room_acoustic_largedataset/
  ```
* Place the **EDC ground truth files** (`.npy`) inside:

  ```
  dataset/room_acoustic_largedataset/EDC/
  ```

Each row in the CSV corresponds to:

```
[room_id, length, width, height, absorption_band_1, ..., absorption_band_7, src_x, src_y, src_z, rec_x, rec_y, rec_z]
```

### Step 2 – Run Training

```bash
python training/train_edcModelPytorchLighteningICASSP_V1.py
```

### Step 3 – Outputs

* The trained model checkpoint will be saved under:

  ```
  Results/{date-time}/Trained_Models/best_model.ckpt
  ```
* Feature and target **scalers** will be saved in the same folder.
* Training and validation loss curves are automatically logged.

### Training Parameters (default)

* Model: **LSTM → Dense(2048) → Output**
* Hidden units: **128**
* Dropout: **0.3**
* Optimizer: **Adam (lr=0.001)**
* Loss: **MSE + custom temporal decay loss**
* Early stopping: **10 epochs patience**
* Max epochs: **200**

---

## 📊 Outputs

After training or inference, you’ll get:

1. **Plots**

   * EDC prediction vs. ground truth
   * RIR waveform reconstruction
   * FFT magnitude spectrum

2. **Audio Files**

   * `predicted_rir.wav` – Predicted RIR reconstruction
   * `actual_rir.wav` – Ground truth RIR reconstruction

3. **Metrics**

   * Mean Squared Error (MSE)
   * Optionally: correlation, log-spectral distance (LSD), SI-SDR, etc.

---

## 📌 Notes

* The provided dataset and checkpoints are examples; you can train your own using PyTorch Lightning.
* Default inference runs on **GPU (if available)**, otherwise CPU.
* Parameters like `stickiness` (default = 0.9) can be tuned to balance low-frequency coherence vs. randomness.

---

## 🛠️ Citation

If you use this work, please cite:

```
@inproceedings{tui2026edc,
    title={ROOM IMPULSE RESPONSE PREDICTION WITH NEURAL NETWORKS: FROM ENERGY DECAY CURVES TO PERCEPTUAL VALIDATION},
    author={Imran Muhammad, Gerald Schuller},
    booktitle={2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    year={2026},
    organization={IEEE}
    }
```
