"""Load a trained RhythmCNN and predict the rhythm class for a signal.

Returns None gracefully if no checkpoint exists yet, so callers can degrade
without a trained model.
"""

import os
import sys

import numpy as np
import torch
from scipy.signal import resample

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rhythm_cnn import RhythmCNN, CLASSES, CLASS_NAMES

torch.backends.cudnn.enabled = False  # GB10 cuDNN workaround

_DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rhythm_cnn.pt")
_CACHE = {}


def load(ckpt_path=_DEFAULT_CKPT):
    if not os.path.exists(ckpt_path):
        return None
    if ckpt_path not in _CACHE:
        ck = torch.load(ckpt_path, map_location="cpu")
        model = RhythmCNN(n_classes=len(ck["classes"]))
        model.load_state_dict(ck["state_dict"])
        model.eval()
        _CACHE[ckpt_path] = (model, ck)
    return _CACHE[ckpt_path]


def predict(mv, fs, ckpt_path=_DEFAULT_CKPT):
    """Return {label, name, prob} or None if no trained model is available."""
    loaded = load(ckpt_path)
    if loaded is None:
        return None
    model, ck = loaded
    target_fs, seconds = ck["fs"], ck["seconds"]
    length = int(target_fs * seconds)

    sig = np.asarray(mv, dtype=np.float32)
    if fs != target_fs and len(sig) > 1:
        sig = resample(sig, int(len(sig) * target_fs / fs)).astype(np.float32)
    if len(sig) >= length:
        start = (len(sig) - length) // 2
        sig = sig[start:start + length]
    else:
        sig = np.pad(sig, (0, length - len(sig)))
    std = sig.std()
    sig = (sig - sig.mean()) / (std if std > 1e-6 else 1.0)

    with torch.inference_mode():
        probs = torch.softmax(model(torch.from_numpy(sig)[None, None, :]), dim=1)[0]
    idx = int(probs.argmax())
    code = ck["classes"][idx]
    return {"label": code, "name": CLASS_NAMES.get(code, code), "prob": float(probs[idx])}
