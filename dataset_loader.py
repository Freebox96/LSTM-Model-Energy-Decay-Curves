import os
import numpy as np
import torch
from torch.utils.data import Dataset

class RIRDataset(Dataset):
    def __init__(self, base_dir, band=0):
        self.base = base_dir
        self.band = band
        self.ids = np.load(os.path.join(base_dir, "ids.npy"))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        rid = self.ids[idx]

        x = np.load(f"{self.base}/X/{rid}.npy")        # (16,)
        y = np.load(f"{self.base}/Y/{rid}.npy")        # (T, 7)

        y = y[:, self.band]                            # single band
        y = np.log(np.maximum(y, 1e-8))                # log-energy

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
