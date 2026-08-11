#!/usr/bin/env python3
"""Download the PhysioNet/CinC Challenge 2017 single-lead ECG set and convert
its records into the CSV layout tools/render_single_lead.py expects.

Purpose: give the single-lead PULSE pipeline a real, labeled dataset for
system validation without waiting for Frontier X Plus device data.

Output (under --out-dir):
  signals/<id>.csv   one column of millivolts, header "mv"
  labels.csv         id, signal_path, answer   (ready for build-dataset/prep)

The 2017 records are single-lead, 300 Hz. Labels: N=normal, A=atrial
fibrillation, O=other rhythm, ~=noisy.

Examples:
  # small sample set (few records), good for a quick smoke test
  python tools/fetch_physionet2017.py --zip-name sample2017.zip --out-dir data/physionet2017

  # full training set, limit how many records to convert
  python tools/fetch_physionet2017.py --limit 200 --out-dir data/physionet2017

  # convert from an already-extracted folder (no download)
  python tools/fetch_physionet2017.py --source-dir /path/with/mat_hea --out-dir data/physionet2017
"""

import argparse
import csv
import os
import urllib.request
import zipfile

import numpy as np
from scipy.io import loadmat

BASE_URL = "https://physionet.org/files/challenge-2017/1.0.0"
LABEL_TEXT = {
    "N": "Sinus rhythm (normal ECG).",
    "A": "Atrial fibrillation.",
    "O": "Other rhythm or abnormality.",
    "~": "Noisy recording; interpretation not possible.",
}


def download_and_extract(zip_name, work_dir):
    """Download <zip_name> from PhysioNet and extract into work_dir. Returns
    the directory that directly contains the .mat/.hea files."""
    os.makedirs(work_dir, exist_ok=True)
    zip_path = os.path.join(work_dir, zip_name)
    if not os.path.exists(zip_path):
        url = f"{BASE_URL}/{zip_name}"
        print(f"downloading {url} ...")
        urllib.request.urlretrieve(url, zip_path)
    print(f"extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(work_dir)
    # The zip usually extracts into a subfolder (e.g. training2017/).
    for root, _dirs, files in os.walk(work_dir):
        if any(f.endswith(".mat") for f in files):
            return root
    raise RuntimeError("no .mat files found after extraction")


def read_gain(hea_path):
    """Parse ADC gain (adu per mV) from a WFDB .hea; default 1000 if absent."""
    try:
        with open(hea_path) as f:
            lines = [ln for ln in f if ln.strip() and not ln.startswith("#")]
        # signal spec line 2, 3rd field like "1000/mV" or "1000(0)/mV"
        field = lines[1].split()[2]
        gain = field.split("/")[0].split("(")[0]
        return float(gain) or 1000.0
    except Exception:
        return 1000.0


def load_reference(data_dir):
    """Return {record_id: label_code} from REFERENCE.csv if present."""
    for name in ("REFERENCE.csv", "REFERENCE-v3.csv", "REFERENCE-original.csv"):
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            ref = {}
            with open(path, newline="") as f:
                for row in csv.reader(f):
                    if len(row) >= 2:
                        ref[row[0]] = row[1]
            return ref
    return {}


def convert(data_dir, out_dir, limit):
    sig_dir = os.path.join(out_dir, "signals")
    os.makedirs(sig_dir, exist_ok=True)
    ref = load_reference(data_dir)

    mats = sorted(f for f in os.listdir(data_dir) if f.endswith(".mat"))
    if limit:
        mats = mats[:limit]

    rows = []
    for fname in mats:
        rec_id = os.path.splitext(fname)[0]
        val = loadmat(os.path.join(data_dir, fname))["val"].astype(np.float64).ravel()
        gain = read_gain(os.path.join(data_dir, rec_id + ".hea"))
        mv = val / gain

        with open(os.path.join(sig_dir, rec_id + ".csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["mv"])
            for v in mv:
                w.writerow([f"{v:.5f}"])

        code = ref.get(rec_id, "")
        rows.append({
            "id": rec_id,
            "signal_path": rec_id + ".csv",
            "answer": LABEL_TEXT.get(code, "Unknown/unlabeled."),
            "label_code": code,
        })

    labels_path = os.path.join(out_dir, "labels.csv")
    with open(labels_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "signal_path", "answer", "label_code"])
        w.writeheader()
        w.writerows(rows)

    print(f"converted {len(rows)} records")
    print(f"  signals: {sig_dir}")
    print(f"  labels:  {labels_path}")
    print("next (render + build labeled dataset), fs=300:")
    print(f"  python tools/render_single_lead.py build-dataset --type raw \\")
    print(f"    --manifest {labels_path} --signal-root {sig_dir} \\")
    print(f"    --image-out {out_dir}/images --json-out {out_dir}/data.json --fs 300 --unit mv")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--zip-name", default="training2017.zip",
                   help="training2017.zip (full) or sample2017.zip (small)")
    p.add_argument("--source-dir", default="",
                   help="use an already-extracted folder of .mat/.hea instead of downloading")
    p.add_argument("--limit", type=int, default=0, help="max records to convert (0 = all)")
    args = p.parse_args()

    data_dir = args.source_dir or download_and_extract(
        args.zip_name, os.path.join(args.out_dir, "_download"))
    convert(data_dir, args.out_dir, args.limit)


if __name__ == "__main__":
    main()
