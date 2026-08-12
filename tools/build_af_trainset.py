#!/usr/bin/env python3
"""Build an AF-focused rhythm training set from MIT-BIH AFDB records.

Segments each record into fixed windows, labels them AF ('A') or non-AF ('N')
from the rhythm annotations, resamples to the RhythmCNN training rate, and writes
labels.csv + signals/*.csv (compatible with models/ecg_dataset.py).

    python tools/build_af_trainset.py --afdb-dir data/validation/afdb --out-dir data/af_train
"""

import argparse
import csv
import glob
import os

import numpy as np
import wfdb
from scipy.signal import resample


def _af_timeline(ann, n):
    flags = np.zeros(n, dtype=bool)
    marks = sorted(zip(ann.sample, [a.strip("\x00") for a in ann.aux_note]))
    state = False
    spans = []
    for samp, note in marks:
        if note.startswith("("):
            state = note[1:].upper().startswith("AFIB")
        spans.append((samp, state))
    for i, (samp, val) in enumerate(spans):
        end = spans[i + 1][0] if i + 1 < len(spans) else n
        flags[samp:end] = val
    return flags


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--afdb-dir", default="data/validation/afdb")
    p.add_argument("--out-dir", default="data/af_train")
    p.add_argument("--target-fs", type=float, default=300.0)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--step", type=float, default=30.0)
    p.add_argument("--max-per-class-per-record", type=int, default=60)
    args = p.parse_args()

    sig_dir = os.path.join(args.out_dir, "signals")
    os.makedirs(sig_dir, exist_ok=True)
    out_len = int(args.target_fs * args.seconds)
    recs = sorted({os.path.splitext(os.path.basename(p_))[0]
                   for p_ in glob.glob(os.path.join(args.afdb_dir, "*.hea"))})

    rows = []
    counts = {"A": 0, "N": 0}
    for rec in recs:
        path = os.path.join(args.afdb_dir, rec)
        try:
            sig, fields = wfdb.rdsamp(path)
            ann = wfdb.rdann(path, "atr")
        except Exception as exc:
            print(f"{rec}: skip ({exc})")
            continue
        fs = float(fields["fs"])
        x = sig[:, 0]
        af = _af_timeline(ann, len(x))
        step = int(args.step * fs)
        win = int(args.seconds * fs)
        per_rec = {"A": 0, "N": 0}
        for s in range(0, len(x) - win, step):
            seg = x[s:s + win]
            frac = af[s:s + win].mean()
            if frac > 0.7:
                code = "A"
            elif frac < 0.05:
                code = "N"
            else:
                continue
            if per_rec[code] >= args.max_per_class_per_record:
                continue
            per_rec[code] += 1
            seg = resample(seg, out_len).astype(np.float32) if fs != args.target_fs else seg
            rid = f"{rec}_{s}_{code}"
            with open(os.path.join(sig_dir, rid + ".csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["mv"])
                for v in seg:
                    w.writerow([f"{float(v):.4f}"])
            rows.append({"id": rid, "signal_path": rid + ".csv", "label_code": code})
            counts[code] += 1
        print(f"{rec}: +{per_rec['A']} AF, +{per_rec['N']} non-AF")

    with open(os.path.join(args.out_dir, "labels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "signal_path", "label_code"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} windows -> {args.out_dir}  (AF {counts['A']}, non-AF {counts['N']})")


if __name__ == "__main__":
    main()
