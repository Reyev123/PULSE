"""1-D CNN for single-lead ECG rhythm classification.

Four classes matching the PhysioNet/CinC 2017 single-lead challenge:
    N = normal sinus, A = atrial fibrillation, O = other rhythm, ~ = noisy.
"""

import torch
import torch.nn as nn

CLASSES = ["N", "A", "O", "~"]
CLASS_NAMES = {"N": "normal sinus", "A": "atrial fibrillation",
               "O": "other rhythm", "~": "noisy"}


class _ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=7, pool=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, padding=k // 2),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv1d(c_out, c_out, k, padding=k // 2),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool),
        )

    def forward(self, x):
        return self.net(x)


class RhythmCNN(nn.Module):
    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            _ConvBlock(1, 32),
            _ConvBlock(32, 64),
            _ConvBlock(64, 128),
            _ConvBlock(128, 128),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):  # x: [B, 1, L]
        return self.head(self.features(x))
