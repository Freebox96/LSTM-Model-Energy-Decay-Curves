import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class MultiBandEDCDataset(Dataset):
    def __init__(self,
                 feature_csv,
                 edc_multiband_dir):
        """
        feature_csv: path to roomFeaturesDataset.csv
        edc_multiband_dir: directory containing *_multiband.npy files
        """
        self.features_df = pd.read_csv(feature_csv)
        self.edc_dir = edc_multiband_dir

        # Build index
        self.samples = []
        for fname in os.listdir(self.edc_dir):
            if fname.endswith("_multiband.npy"):
                rir_id = fname.replace("_multiband.npy", "")
                self.samples.append((rir_id, fname))

        print(f"Loaded {len(self.samples)} multi-band EDC samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rir_id, edc_file = self.samples[idx]

        # -----------------------
        # Load multi-band EDC
        # Shape: (T, 7)
        # -----------------------
        edc = np.load(os.path.join(self.edc_dir, edc_file))
        edc = torch.tensor(edc, dtype=torch.float32)

        # -----------------------
        # Load room features
        # -----------------------
        row = self.features_df[self.features_df["RIR_ID"] == rir_id]
        if row.empty:
            raise ValueError(f"Room features not found for {rir_id}")

        features = row.drop(columns=["RIR_ID"]).values.squeeze()
        features = torch.tensor(features, dtype=torch.float32)

        return features, edc
