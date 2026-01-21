# Multi-band EDC Prediction with LSTM

This project implements a multi-band dataset pipeline and LSTM model for predicting Energy Decay Curves (EDCs) from room acoustic features.

## Workflow

1. Preprocess multiband EDC dataset
2. Build final dataset
3. Train baseline single-band LSTM
4. Train multi-output multi-band LSTM
5. Evaluate using:
   - band-wise RT30 proxy
   - band-wise clarity proxy (C50-like)
   - spectral envelope similarity

## Files

- `train_baseline.py` : single-band baseline
- `train_multiband.py` : multi-output LSTM (7 bands)
- `evaluate_multiband.py` : metrics evaluation
- `plot_paper_figures.py` : generates plots like paper Fig 4/5/6 + Table 2

## Note

Datasets are excluded from GitHub using `.gitignore`.
