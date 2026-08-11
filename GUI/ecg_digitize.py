"""Best-effort digitization of a single-lead ECG image into a signal.

EXPERIMENTAL: works on clean, single-trace images. It extracts the dark
trace column-by-column and calibrates time/amplitude from the red grid when
detectable (1 small box = 0.04 s = 0.1 mV); otherwise it falls back to the
provided duration for the time axis. 12-lead layouts, heavy noise, or missing
grids will degrade or fail.
"""

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks


def _detect_grid_spacing(rgb):
    """Pixels between red gridlines (a small box), or None."""
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    reddish = (R > 150) & (R - G > 25) & (R - B > 25)
    col = reddish.sum(axis=0).astype(float)
    if col.max() < 1:
        return None
    col -= col.mean()
    ac = np.correlate(col, col, mode="full")[len(col) - 1:]
    if ac[0] <= 0:
        return None
    peaks, _ = find_peaks(ac, height=0.3 * ac[0], distance=3)
    peaks = peaks[peaks > 3]
    return float(peaks[0]) if len(peaks) else None


def digitize(pil_image, seconds=10.0):
    """Return {'ok', 'mv', 'fs', 'grid_detected'} or {'ok': False, 'reason'}."""
    rgb = np.asarray(pil_image.convert("RGB"), dtype=float)
    h, w, _ = rgb.shape
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    dark = lum < 100  # ECG trace is near-black; grid is light red

    # Follow the trace by picking, per column, the dark pixel nearest the
    # previous column's position (avoids jumping to frame/axis pixels).
    ys = np.full(w, np.nan)
    prev = None
    for x in range(w):
        rows = np.where(dark[:, x])[0]
        if len(rows) == 0:
            continue
        y = np.median(rows) if prev is None else rows[np.argmin(np.abs(rows - prev))]
        ys[x] = y
        prev = y

    valid = ~np.isnan(ys)
    if valid.sum() < 0.3 * w:
        return {"ok": False, "reason": "no clear single-lead trace detected"}
    ys = np.interp(np.arange(w), np.flatnonzero(valid), ys[valid])

    box = _detect_grid_spacing(rgb)
    signal = (h - ys)                       # flip: image row 0 is top
    # Remove baseline wander/offset with a wide rolling median so the
    # isoelectric line sits at 0 (no spurious large negative excursions).
    pps = w / float(seconds or 10.0)
    k = max(3, int(0.7 * pps) | 1)          # ~0.7 s window, odd
    signal = signal - median_filter(signal, size=k)
    if box and box > 2:
        fs = 1.0 / (0.04 / box)             # 1 small box = 0.04 s
        mv = signal * (0.1 / box)           # 1 small box = 0.1 mV
        grid = True
    else:
        fs = w / float(seconds or 10.0)     # assume image width == duration
        mv = signal / max(1.0, 0.2 * h)     # nominal amplitude scaling
        grid = False

    # Orient so the dominant deflection (QRS) points up, like a standard EKG.
    hi, lo = np.percentile(mv, 99.5), np.percentile(mv, 0.5)
    if abs(lo) > abs(hi):
        mv = -mv

    return {"ok": True, "mv": mv.astype(float), "fs": float(fs), "grid_detected": grid}
