"""Parse uploaded raw single-lead signals and render them to ECG images.

Reuses the rendering logic from tools/render_single_lead.py so the GUI, the
batch harness, and training all produce visually identical ECG images.
"""

import csv
import io
import os
import sys
import tempfile

import numpy as np
from PIL import Image

# Make tools/render_single_lead.py importable.
_TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

import render_single_lead as rsl  # noqa: E402

SIGNAL_EXTS = (".csv", ".txt", ".npy")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tif", ".tiff")


def is_signal(filename):
    return os.path.splitext(filename)[1].lower() in SIGNAL_EXTS


def is_image(filename):
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTS


def parse_signal_bytes(data, filename, column=0, delimiter=",", skip_header=1):
    """Parse a 1-D signal from uploaded bytes (csv/txt/npy)."""
    if os.path.splitext(filename)[1].lower() == ".npy":
        arr = np.load(io.BytesIO(data))
        return np.asarray(arr, dtype=np.float64).ravel()

    text = data.decode("utf-8", "ignore")
    values = []
    for i, row in enumerate(csv.reader(io.StringIO(text), delimiter=delimiter)):
        if i < skip_header or not row:
            continue
        try:
            values.append(float(row[column]))
        except (ValueError, IndexError):
            continue
    if not values:
        raise ValueError("No numeric samples parsed; check column/delimiter/header.")
    return np.asarray(values, dtype=np.float64)


def signal_to_mv(sig, unit="mv"):
    return rsl.to_millivolts(sig, unit)


def render_ecg_png(sig_mv, fs, seconds=10.0, start=0.0, dpi=200):
    """Render a millivolt signal to an ECG-style PNG; return bytes."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = tmp.name
    try:
        rsl.render(sig_mv, fs, path, seconds=seconds, start=start, dpi=dpi)
        with open(path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(path):
            os.remove(path)


def image_bytes_to_pil(image_bytes):
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")
