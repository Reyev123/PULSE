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


def _augment(sig, fs, rng):
    """Randomly add baseline-wander / powerline / EMG-like noise at a random SNR."""
    if rng.random() < 0.4:  # keep ~40% clean
        return sig
    rms = float(np.sqrt(np.mean(sig ** 2))) or 1.0
    snr = rng.uniform(0.0, 18.0)
    t = np.arange(len(sig)) / fs
    kind = rng.integers(0, 3)
    if kind == 0:
        noise = np.sin(2 * np.pi * 0.3 * t) + 0.5 * np.sin(2 * np.pi * 0.15 * t)
    elif kind == 1:
        noise = np.sin(2 * np.pi * 50.0 * t)
    else:
        noise = rng.standard_normal(len(sig))
    noise = noise.astype(np.float32)
    noise *= (rms / (10 ** (snr / 20.0))) / (float(np.sqrt(np.mean(noise ** 2))) or 1.0)
    return sig + noise


class ECGRhythmDataset(Dataset):
    def __init__(self, data_dir, fs=300, seconds=30, limit=0, augment=False):
        self.fs = fs
        self.length = int(fs * seconds)
        self.augment = augment
        self.rng = np.random.default_rng(0)
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
        if self.augment:
            sig = _augment(sig, self.fs, self.rng)
        std = sig.std()
        sig = (sig - sig.mean()) / (std if std > 1e-6 else 1.0)  # z-score
        return torch.from_numpy(sig).unsqueeze(0), label
