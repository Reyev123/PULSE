"""Dataset for single-lead ECG rhythm classification.

Reads a labels.csv (columns: id, signal_path, label_code) and the matching
per-record CSV signals (single 'mv' column), producing fixed-length windows.
Compatible with the output of tools/fetch_physionet2017.py.
"""

import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from rhythm_cnn import CLASSES

_LABEL_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def _load_signal(path):
    vals = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header 'mv'
        for row in reader:
            if row:
                try:
                    vals.append(float(row[0]))
                except ValueError:
                    pass
    return np.asarray(vals, dtype=np.float32)


def _fix_length(sig, length):
    """Pad (zeros) or center-crop to a fixed length."""
    if len(sig) >= length:
        start = (len(sig) - length) // 2
        return sig[start:start + length]
    out = np.zeros(length, dtype=np.float32)
    out[: len(sig)] = sig
    return out


class ECGRhythmDataset(Dataset):
    def __init__(self, data_dir, fs=300, seconds=30, limit=0):
        self.length = int(fs * seconds)
        self.signals_dir = os.path.join(data_dir, "signals")
        self.items = []
        with open(os.path.join(data_dir, "labels.csv"), newline="") as f:
            for row in csv.DictReader(f):
                code = (row.get("label_code") or "").strip()
                if code in _LABEL_TO_IDX:
                    self.items.append((row["signal_path"], _LABEL_TO_IDX[code]))
        if limit:
            self.items = self.items[:limit]
        if not self.items:
            raise ValueError("no labeled items found (need label_code in N/A/O/~)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rel, label = self.items[idx]
        sig = _load_signal(os.path.join(self.signals_dir, rel))
        sig = _fix_length(sig, self.length)
        std = sig.std()
        sig = (sig - sig.mean()) / (std if std > 1e-6 else 1.0)  # z-score
        return torch.from_numpy(sig).unsqueeze(0), label
