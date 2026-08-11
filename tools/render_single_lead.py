#!/usr/bin/env python3
"""Render single-lead ECG signals (e.g. Frontier X Plus exports) into
clinical-style ECG images for the GUI.

The `render` sub-command draws one signal file to a PNG on a standard ECG grid
(25 mm/s, 10 mm/mV). The same render()/to_millivolts() helpers are imported by
GUI/signal_io.py.
"""

import argparse
import csv
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# --- signal loading ---------------------------------------------------------
def load_signal(path, column, delimiter, skip_header):
    """Load a 1-D signal from a CSV/TXT file.

    Frontier X Plus exports are typically CSV with a voltage column; adjust
    --column / --delimiter to match your export.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
        return np.asarray(arr, dtype=np.float64).ravel()

    values = []
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for i, row in enumerate(reader):
            if i < skip_header or not row:
                continue
            try:
                values.append(float(row[column]))
            except (ValueError, IndexError):
                continue
    if not values:
        raise ValueError(f"No numeric samples parsed from {path}")
    return np.asarray(values, dtype=np.float64)


def to_millivolts(sig, unit):
    """Normalize raw samples to millivolts."""
    if unit == "mv":
        return sig
    if unit == "uv":
        return sig / 1000.0
    if unit == "raw":
        # Unknown ADC scale: center and scale to a ~1 mV typical amplitude.
        sig = sig - np.median(sig)
        peak = np.percentile(np.abs(sig), 99) or 1.0
        return sig / peak
    raise ValueError(f"unknown unit {unit}")


# --- rendering --------------------------------------------------------------
def render(sig_mv, fs, out_path, seconds=10.0, start=0.0, mm_per_s=25.0,
           mm_per_mv=10.0, dpi=200):
    """Render a single-lead window [start, start+seconds] onto ECG graph paper."""
    i0 = max(0, int(start * fs))
    n = int(seconds * fs)
    sig_mv = sig_mv[i0:i0 + n]
    t = np.arange(len(sig_mv)) / fs

    # Physical size: time axis width from paper speed, amplitude clamped so a
    # normal QRS fits the strip height.
    width_in = (seconds * mm_per_s) / 25.4
    height_in = (30.0 * 1.0) / 25.4 * (mm_per_mv / 10.0) * 1.0
    height_in = max(height_in, 1.4)

    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=dpi)

    ax.plot(t, sig_mv, color="black", linewidth=0.8)

    # ECG grid: minor 0.04 s / 0.1 mV, major 0.2 s / 0.5 mV.
    ax.xaxis.set_minor_locator(MultipleLocator(0.04))
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.grid(which="minor", color="#f4b8b8", linewidth=0.4)
    ax.grid(which="major", color="#e57373", linewidth=0.7)

    ax.set_xlim(0, seconds)
    y_span = 1.5 * (10.0 / mm_per_mv)
    ax.set_ylim(-y_span, y_span)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_color("#e57373")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout(pad=0.1)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)


# --- sub-command ------------------------------------------------------------
def cmd_render(args):
    sig = to_millivolts(
        load_signal(args.input, args.column, args.delimiter, args.skip_header),
        args.unit,
    )
    render(sig, args.fs, args.output, seconds=args.seconds, dpi=args.dpi)
    print(f"wrote {args.output}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    # shared signal-parsing options
    def add_signal_opts(sp):
        sp.add_argument("--fs", type=float, default=500.0,
                        help="sampling rate in Hz (Frontier X Plus is often 500)")
        sp.add_argument("--column", type=int, default=0,
                        help="0-based column index of the voltage samples")
        sp.add_argument("--delimiter", default=",")
        sp.add_argument("--skip-header", type=int, default=1)
        sp.add_argument("--unit", choices=["mv", "uv", "raw"], default="mv")
        sp.add_argument("--seconds", type=float, default=10.0)
        sp.add_argument("--dpi", type=int, default=200)

    r = sub.add_parser("render", help="render one signal to a PNG")
    r.add_argument("--input", required=True)
    r.add_argument("--output", required=True)
    add_signal_opts(r)
    r.set_defaults(func=cmd_render)

    return p


if __name__ == "__main__":
    parser = build_parser()
    ns = parser.parse_args()
    ns.func(ns)
