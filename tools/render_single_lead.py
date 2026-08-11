#!/usr/bin/env python3
"""Render single-lead ECG signals (e.g. Frontier X Plus exports) into
clinical-style ECG images and build a LLaVA/PULSE fine-tuning dataset.

Two sub-commands:

  render        Render one signal file to a single PNG (for quick inspection
                or ad-hoc inference with run_llava.py).

  build-dataset Render many signals listed in a manifest CSV and emit a
                train.json in the LLaVA conversation format that
                train_mem.py expects.

The renderer draws a standard ECG grid (25 mm/s, 10 mm/mV, red gridlines) so
the produced image resembles the 12-lead printouts PULSE was trained on as
closely as a single lead allows. This narrows — but does not remove — the
domain gap; you still need to fine-tune (see finetune_pulse_singlelead_lora.sh).
"""

import argparse
import csv
import json
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
def render(sig_mv, fs, out_path, seconds=10.0, mm_per_s=25.0, mm_per_mv=10.0,
           dpi=200):
    """Render a single lead onto ECG graph paper and save a PNG."""
    n = int(seconds * fs)
    sig_mv = sig_mv[:n] if len(sig_mv) >= n else sig_mv
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


# --- LLaVA dataset format ---------------------------------------------------
DEFAULT_PROMPT = "Please write a clinical report based on this single-lead ECG image."


def make_conversation(prompt, answer):
    """Return a LLaVA-format conversation with the <image> token."""
    return [
        {"from": "human", "value": "<image>\n" + prompt},
        {"from": "gpt", "value": answer},
    ]


# --- sub-commands -----------------------------------------------------------
def cmd_render(args):
    sig = to_millivolts(
        load_signal(args.input, args.column, args.delimiter, args.skip_header),
        args.unit,
    )
    render(sig, args.fs, args.output, seconds=args.seconds, dpi=args.dpi)
    print(f"wrote {args.output}")


def cmd_build_dataset(args):
    """Read a manifest CSV and emit images + train.json.

    Manifest columns (header required): id, answer
    Plus one of: signal_path (--type raw) or image_path (--type images)
    Optional column: prompt  (falls back to --prompt / DEFAULT_PROMPT)
    Paths are resolved relative to --signal-root.
    """
    if args.type == "raw":
        if not args.image_out:
            raise ValueError("--image-out is required for --type raw")
        os.makedirs(args.image_out, exist_ok=True)
    records = []
    with open(args.manifest, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        if args.type == "raw":
            sig = to_millivolts(
                load_signal(os.path.join(args.signal_root, row["signal_path"]),
                            args.column, args.delimiter, args.skip_header),
                args.unit,
            )
            rel_img = f"{row['id']}.png"
            render(sig, args.fs, os.path.join(args.image_out, rel_img),
                   seconds=args.seconds, dpi=args.dpi)
        else:
            rel_img = row["image_path"]
        records.append({
            "id": row["id"],
            "image": rel_img,
            "conversations": make_conversation(
                row.get("prompt") or args.prompt, row["answer"]),
        })

    with open(args.json_out, "w") as f:
        json.dump(records, f, indent=2)
    where = args.image_out if args.type == "raw" else args.signal_root
    print(f"wrote {len(records)} records to {args.json_out} (images at {where})")


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
SIGNAL_EXTS = (".csv", ".txt", ".npy")


def cmd_prep(args):
    """Build a label-free inference manifest for batch inference.

    Two input types:
      images  --input is a folder of already-rendered ECG images.
      raw     --input is a folder of Frontier X Plus signal files; each is
              rendered to --image-out.

    A placeholder assistant turn is included because model_ecg_resume.py
    requires conversations[1]; it is ignored during pure inference.
    """
    files = sorted(f for f in os.listdir(args.input)
                   if os.path.splitext(f)[1].lower() in
                   (IMAGE_EXTS if args.type == "images" else SIGNAL_EXTS))
    if not files:
        raise ValueError(f"no {args.type} files found in {args.input}")

    if args.type == "raw":
        os.makedirs(args.image_out, exist_ok=True)

    records = []
    for fname in files:
        stem = os.path.splitext(fname)[0]
        if args.type == "images":
            rel_img = fname
        else:
            sig = to_millivolts(
                load_signal(os.path.join(args.input, fname), args.column,
                            args.delimiter, args.skip_header),
                args.unit)
            rel_img = f"{stem}.png"
            render(sig, args.fs, os.path.join(args.image_out, rel_img),
                   seconds=args.seconds, dpi=args.dpi)
        records.append({
            "id": stem,
            "image": rel_img,
            "conversations": [
                {"from": "human", "value": "<image>\n" + args.prompt},
                {"from": "gpt", "value": ""},  # placeholder; ignored at inference
            ],
        })

    with open(args.json_out, "w") as f:
        json.dump(records, f, indent=2)
    image_folder = args.input if args.type == "images" else args.image_out
    print(f"wrote {len(records)} records to {args.json_out}")
    print(f"image-folder: {image_folder}")


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

    b = sub.add_parser("build-dataset", help="build labeled train.json from images or raw signals")
    b.add_argument("--manifest", required=True,
                   help="CSV: id, answer, (signal_path|image_path)[, prompt]")
    b.add_argument("--type", choices=["images", "raw"], default="raw")
    b.add_argument("--signal-root", default=".",
                   help="root for signal_path/image_path in the manifest")
    b.add_argument("--image-out", default="",
                   help="where to write rendered images (--type raw only)")
    b.add_argument("--json-out", required=True)
    b.add_argument("--prompt", default=DEFAULT_PROMPT)
    add_signal_opts(b)
    b.set_defaults(func=cmd_build_dataset)

    pr = sub.add_parser("prep",
                        help="label-free inference manifest from images or raw signals")
    pr.add_argument("--input", required=True,
                    help="folder of images (--type images) or signals (--type raw)")
    pr.add_argument("--type", choices=["images", "raw"], required=True)
    pr.add_argument("--image-out", default="",
                    help="where to write rendered images (--type raw only)")
    pr.add_argument("--json-out", required=True)
    pr.add_argument("--prompt", default=DEFAULT_PROMPT)
    add_signal_opts(pr)
    pr.set_defaults(func=cmd_prep)

    return p


if __name__ == "__main__":
    parser = build_parser()
    ns = parser.parse_args()
    ns.func(ns)
