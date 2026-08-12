#!/usr/bin/env python3
"""Download PhysioNet validation databases via wfdb.

These stand in for Frontier X Plus (single-lead, chest, exercise, motion) since
no public Frontier X Plus dataset exists:
  mitdb   MIT-BIH Arrhythmia      — per-beat PVC/PAC/HR ground truth
  svdb    MIT-BIH Supraventricular— extra PAC/SVEB examples
  afdb    MIT-BIH Atrial Fib      — AF rhythm ground truth
  nstdb   Noise Stress Test       — motion/EMG artifact (the "hard cases")
  incartdb St Petersburg INCART   — 12-lead Holter beat truth (use one lead)

Usage:
    python tools/fetch_validation_sets.py --dbs mitdb afdb nstdb
    python tools/fetch_validation_sets.py --dbs mitdb --all
"""

import argparse
import os

import wfdb

# Small, representative default subsets (mix of PVC/PAC/AF/noise).
DEFAULTS = {
    "mitdb": ["100", "106", "107", "119", "200", "203", "208", "209", "223", "233"],
    "svdb": ["800", "801", "803", "805", "806"],
    "afdb": ["04015", "04043", "04126", "05121", "06426", "07879", "08378", "08455"],
    "nstdb": ["118e24", "118e12", "118e06", "118e00", "119e24", "119e06", "em", "ma", "bw"],
    "incartdb": ["I01", "I02", "I03"],
    "ltafdb": ["00", "01", "03", "05", "07", "08"],  # independent AF test (not in training)
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dbs", nargs="+", default=["mitdb", "afdb", "nstdb"],
                   choices=list(DEFAULTS))
    p.add_argument("--out-dir", default="data/validation")
    p.add_argument("--all", action="store_true", help="download the full database")
    args = p.parse_args()

    for db in args.dbs:
        dest = os.path.join(args.out_dir, db)
        os.makedirs(dest, exist_ok=True)
        records = None if args.all else DEFAULTS[db]
        print(f"downloading {db} -> {dest} ({'all' if records is None else len(records)} records)")
        try:
            wfdb.dl_database(db, dest, records=records)
        except Exception as exc:
            print(f"  ! {db} failed: {exc}")
            continue
        print(f"  done: {db}")

    print("\nnext:\n  python tools/validate_system.py beats --data-dir data/validation/mitdb\n"
          "  python tools/validate_system.py af    --data-dir data/validation/afdb\n"
          "  python tools/validate_system.py noise --data-dir data/validation/mitdb")


if __name__ == "__main__":
    main()
