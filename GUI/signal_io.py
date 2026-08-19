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
EDF_EXTS = (".edf", ".bdf")


def is_signal(filename):
    return os.path.splitext(filename)[1].lower() in SIGNAL_EXTS


def is_image(filename):
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTS


def is_edf(filename):
    return os.path.splitext(filename)[1].lower() in EDF_EXTS


def _pick_ecg_channel(labels):
    """Index of the best ECG-like signal, skipping EDF+ annotation channels."""
    non_annot = [i for i, lab in enumerate(labels)
                 if "annotation" not in lab.lower()]
    for i in non_annot:
        if "ecg" in labels[i].lower() or "ekg" in labels[i].lower():
            return i
    return non_annot[0] if non_annot else 0


def parse_edf_bytes(data, channel=None):
    """Read one ECG channel from EDF/EDF+ bytes.

    EDF stores a plain-ASCII header followed by int16 data records. Each record
    holds ``nsamp[i]`` samples per signal, concatenated. We read the whole data
    block once, slice out the chosen channel's columns, and apply the standard
    digital -> physical calibration.

    Returns (signal, fs, unit) where ``unit`` is one of 'mv'/'uv'/'raw' derived
    from the channel's physical dimension so the caller can convert to mV.
    """
    if len(data) < 256:
        raise ValueError("File too small to be a valid EDF.")

    is_bdf = data[1:8] == b"BIOSEMI"
    sample_bytes = 3 if is_bdf else 2

    def txt(a, b):
        return data[a:b].decode("ascii", "replace").strip()

    n_signals = int(txt(252, 256))
    n_records = int(txt(236, 244))
    rec_dur = float(txt(244, 252))
    if n_signals <= 0 or rec_dur <= 0:
        raise ValueError("Malformed EDF header (signals/record duration).")

    # Per-signal header fields are stored column-wise after the 256-byte header.
    base = 256
    labels = [txt(base + i * 16, base + (i + 1) * 16) for i in range(n_signals)]
    base += 16 * n_signals
    base += 80 * n_signals  # transducer
    dims = [txt(base + i * 8, base + (i + 1) * 8) for i in range(n_signals)]
    base += 8 * n_signals
    pmin = [float(txt(base + i * 8, base + (i + 1) * 8)) for i in range(n_signals)]
    base += 8 * n_signals
    pmax = [float(txt(base + i * 8, base + (i + 1) * 8)) for i in range(n_signals)]
    base += 8 * n_signals
    dmin = [float(txt(base + i * 8, base + (i + 1) * 8)) for i in range(n_signals)]
    base += 8 * n_signals
    dmax = [float(txt(base + i * 8, base + (i + 1) * 8)) for i in range(n_signals)]
    base += 8 * n_signals
    base += 80 * n_signals  # prefiltering
    nsamp = [int(txt(base + i * 8, base + (i + 1) * 8)) for i in range(n_signals)]

    ch = _pick_ecg_channel(labels) if channel is None else int(channel)
    if not 0 <= ch < n_signals:
        raise ValueError(f"Channel {ch} out of range (0..{n_signals - 1}).")

    header_bytes = int(txt(184, 192)) or (256 * (n_signals + 1))
    rec_samples = sum(nsamp)
    rec_bytes = rec_samples * sample_bytes
    raw = np.frombuffer(data, dtype=np.uint8, offset=header_bytes,
                        count=n_records * rec_bytes)
    raw = raw.reshape(n_records, rec_bytes)

    off = sum(nsamp[:ch]) * sample_bytes
    col = raw[:, off:off + nsamp[ch] * sample_bytes]
    if is_bdf:  # 24-bit little-endian signed -> int32
        b = col.reshape(-1, 3).astype(np.int32)
        dig = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16))
        dig = np.where(dig >= (1 << 23), dig - (1 << 24), dig)
    else:
        dig = col.reshape(n_records, -1).view("<i2").reshape(-1).astype(np.float64)

    span = (dmax[ch] - dmin[ch]) or 1.0
    gain = (pmax[ch] - pmin[ch]) / span
    sig = (dig.astype(np.float64) - dmin[ch]) * gain + pmin[ch]

    fs = nsamp[ch] / rec_dur
    dim = dims[ch].lower()
    unit = "uv" if dim in ("uv", "µv", "\u00b5v") else "mv" if dim == "mv" else "raw"
    return sig, float(fs), unit


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
