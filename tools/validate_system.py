#!/usr/bin/env python3
"""Validate the single-lead system against annotated PhysioNet databases.

Subcommands:
  beats  Per-beat scoring of the signal analyzer (HR + PVC/PAC) vs WFDB
         annotations (bxb-style matching). Works on mitdb / svdb / incartdb.
  af     RhythmCNN AF detection vs afdb rhythm annotations (30 s windows).
  noise  Robustness sweep: add baseline-wander / powerline / EMG-like noise at
         several SNRs and measure PVC-detection degradation on mitdb.

Prepare data first with tools/fetch_validation_sets.py.
"""

import argparse
import glob
import os
import sys

import numpy as np
import wfdb

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "GUI"))
sys.path.insert(0, os.path.join(_HERE, "..", "models"))
import ecg_analysis as eca  # noqa: E402
try:
    import rhythm_infer  # noqa: E402
except Exception:
    rhythm_infer = None

# WFDB beat symbols: which map to PVC vs PAC (supraventricular) ground truth.
PVC_SYMS = {"V", "E"}
PAC_SYMS = {"A", "a", "J", "S"}
BEAT_SYMS = set("NLRBAaJSVrFejnE/fQ?")


def _record_ids(data_dir):
    return sorted({os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(data_dir, "*.hea"))})


def _match(ref_idx, det_idx, tol):
    """Greedy nearest-neighbour match; returns (pairs, matched_ref, matched_det)."""
    pairs = []
    used = np.zeros(len(det_idx), dtype=bool)
    for ri, r in enumerate(ref_idx):
        if len(det_idx) == 0:
            break
        d = np.abs(det_idx - r)
        j = int(np.argmin(d))
        if d[j] <= tol and not used[j]:
            used[j] = True
            pairs.append((ri, j))
    return pairs


def _ref_class(sym):
    if sym in PVC_SYMS:
        return "PVC"
    if sym in PAC_SYMS:
        return "PAC"
    return "normal"


def cmd_beats(args):
    recs = _record_ids(args.data_dir)
    agg = {"det_tp": 0, "det_fp": 0, "det_fn": 0,
           "PVC": {"tp": 0, "fp": 0, "fn": 0}, "PAC": {"tp": 0, "fp": 0, "fn": 0},
           "hr_abs": [], "n": 0}
    print(f"{'rec':<8}{'fs':>5}{'HRref':>7}{'HRest':>7}"
          f"{'PVCse':>7}{'PVCppv':>8}{'PACse':>7}{'PACppv':>8}")
    for rec in recs:
        path = os.path.join(args.data_dir, rec)
        try:
            sig, fields = wfdb.rdsamp(path)
            ann = wfdb.rdann(path, "atr")
        except Exception as exc:
            print(f"{rec:<8} skip ({exc})")
            continue
        fs = float(fields["fs"])
        lead = 0
        x = sig[:args.limit_samples, lead] if args.limit_samples else sig[:, lead]
        ref_mask = [s in BEAT_SYMS for s in ann.symbol]
        ref_idx = np.array([ann.sample[i] for i in range(len(ann.sample))
                            if ref_mask[i] and (not args.limit_samples or ann.sample[i] < len(x))])
        ref_sym = [ann.symbol[i] for i in range(len(ann.sample))
                   if ref_mask[i] and (not args.limit_samples or ann.sample[i] < len(x))]
        if len(ref_idx) < 3:
            continue
        a = eca.analyze(x, fs)
        if not a.get("ok"):
            print(f"{rec:<8} analyze failed: {a.get('reason')}")
            continue
        det_idx = np.array([b["idx"] for b in a["beats"]])
        det_type = [b["type"] for b in a["beats"]]

        tol = int(0.15 * fs)
        pairs = _match(ref_idx, det_idx, tol)
        matched_ref = {ri for ri, _ in pairs}
        matched_det = {di for _, di in pairs}
        det_tp = len(pairs)
        det_fp = len(det_idx) - len(matched_det)
        det_fn = len(ref_idx) - len(matched_ref)
        agg["det_tp"] += det_tp
        agg["det_fp"] += det_fp
        agg["det_fn"] += det_fn

        rec_cls = {"PVC": {"tp": 0, "fp": 0, "fn": 0}, "PAC": {"tp": 0, "fp": 0, "fn": 0}}
        for ri, di in pairs:
            rc = _ref_class(ref_sym[ri])
            pc = det_type[di]
            for cls in ("PVC", "PAC"):
                if rc == cls and pc == cls:
                    rec_cls[cls]["tp"] += 1
                elif rc == cls and pc != cls:
                    rec_cls[cls]["fn"] += 1
                elif rc != cls and pc == cls:
                    rec_cls[cls]["fp"] += 1
        # ectopic ref beats we never detected count as FN for their class
        for ri in range(len(ref_idx)):
            if ri not in matched_ref:
                rc = _ref_class(ref_sym[ri])
                if rc in rec_cls:
                    rec_cls[rc]["fn"] += 1
        for cls in ("PVC", "PAC"):
            for k in ("tp", "fp", "fn"):
                agg[cls][k] += rec_cls[cls][k]

        hr_ref = 60.0 / np.median(np.diff(ref_idx) / fs)
        agg["hr_abs"].append(abs(hr_ref - a["heart_rate"]))
        agg["n"] += 1

        def rate(d):
            se = d["tp"] / (d["tp"] + d["fn"]) if d["tp"] + d["fn"] else float("nan")
            ppv = d["tp"] / (d["tp"] + d["fp"]) if d["tp"] + d["fp"] else float("nan")
            return se, ppv
        pse, pppv = rate(rec_cls["PVC"])
        ase, appv = rate(rec_cls["PAC"])
        print(f"{rec:<8}{fs:>5.0f}{hr_ref:>7.0f}{a['heart_rate']:>7.0f}"
              f"{pse:>7.2f}{pppv:>8.2f}{ase:>7.2f}{appv:>8.2f}")

    print("\n=== aggregate ===")
    dt, df, dn = agg["det_tp"], agg["det_fp"], agg["det_fn"]
    print(f"beat detection: Se {dt/(dt+dn):.3f}  PPV {dt/(dt+df):.3f}  "
          f"(TP {dt} FP {df} FN {dn})")
    for cls in ("PVC", "PAC"):
        d = agg[cls]
        se = d["tp"] / (d["tp"] + d["fn"]) if d["tp"] + d["fn"] else float("nan")
        ppv = d["tp"] / (d["tp"] + d["fp"]) if d["tp"] + d["fp"] else float("nan")
        print(f"{cls}: Se {se:.3f}  PPV {ppv:.3f}  (TP {d['tp']} FP {d['fp']} FN {d['fn']})")
    if agg["hr_abs"]:
        print(f"HR MAE: {np.mean(agg['hr_abs']):.1f} bpm over {agg['n']} records")


def _rhythm_timeline(ann, n):
    """Per-sample AF flag from afdb aux_note markers ((AFIB / (N / (AFL / (J)."""
    flags = np.zeros(n, dtype=bool)
    cur = False
    marks = sorted(zip(ann.sample, [a.strip("\x00") for a in ann.aux_note]))
    starts = []
    for samp, note in marks:
        if note.startswith("("):
            cur = note[1:].upper().startswith("AFIB")
        starts.append((samp, cur))
    for i, (samp, val) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else n
        flags[samp:end] = val
    return flags


def cmd_af(args):
    ckpt = args.ckpt or rhythm_infer._DEFAULT_CKPT if rhythm_infer else None
    if rhythm_infer is None or rhythm_infer.load(ckpt) is None:
        print("RhythmCNN checkpoint not found. Train it first.")
        return
    print(f"checkpoint: {ckpt}")
    recs = _record_ids(args.data_dir)
    tp = fp = tn = fn = 0
    win = 30.0
    for rec in recs:
        path = os.path.join(args.data_dir, rec)
        try:
            sig, fields = wfdb.rdsamp(path)
            ann = wfdb.rdann(path, "atr")
        except Exception as exc:
            print(f"{rec}: skip ({exc})")
            continue
        fs = float(fields["fs"])
        x = sig[:, 0]
        af_flag = _rhythm_timeline(ann, len(x))
        step = int(win * fs)
        for s in range(0, len(x) - step, step):
            seg = x[s:s + step]
            ref_af = af_flag[s:s + step].mean() > 0.5
            pred = rhythm_infer.predict(seg, fs, ckpt_path=ckpt)
            pred_af = bool(pred and pred["label"] == "A")
            if ref_af and pred_af:
                tp += 1
            elif ref_af and not pred_af:
                fn += 1
            elif not ref_af and pred_af:
                fp += 1
            else:
                tn += 1
    se = tp / (tp + fn) if tp + fn else float("nan")
    sp = tn / (tn + fp) if tn + fp else float("nan")
    print(f"AF detection (30 s windows): sensitivity {se:.3f}  specificity {sp:.3f}")
    print(f"  TP {tp}  FN {fn}  FP {fp}  TN {tn}")


def _add_noise(x, fs, kind, snr_db, rng, real=None):
    rms = np.sqrt(np.mean(x ** 2)) or 1.0
    t = np.arange(len(x)) / fs
    if real is not None:
        start = rng.integers(0, max(1, len(real) - len(x)))
        noise = np.resize(real[start:start + len(x)], len(x)).astype(float)
    elif kind == "bw":
        noise = np.sin(2 * np.pi * 0.3 * t) + 0.5 * np.sin(2 * np.pi * 0.15 * t)
    elif kind == "pl":
        noise = np.sin(2 * np.pi * 60.0 * t)
    else:  # emg-like broadband
        noise = rng.standard_normal(len(x))
    noise *= (rms / (10 ** (snr_db / 20.0))) / (np.sqrt(np.mean(noise ** 2)) or 1.0)
    return x + noise


def _load_real_noise(noise_dir):
    """Load nstdb noise-only records (em/ma/bw) as 1-D arrays."""
    out = {}
    if not noise_dir:
        return out
    for name in ("em", "ma", "bw"):
        path = os.path.join(noise_dir, name)
        if os.path.exists(path + ".hea"):
            try:
                sig, _ = wfdb.rdsamp(path)
                out[name] = sig[:, 0].astype(float)
            except Exception:
                pass
    return out


def cmd_noise(args):
    recs = [r for r in _record_ids(args.data_dir) if r not in ("em", "ma", "bw")]
    rng = np.random.default_rng(0)
    snrs = [24, 18, 12, 6, 0]
    real = _load_real_noise(args.noise_dir)
    kinds = list(real) if real else ["bw", "pl", "emg"]
    print(f"noise source: {'real nstdb ' + '/'.join(real) if real else 'synthetic'}")
    print(f"{'SNR':>5} {'noise':>6} {'PVCse':>7} {'PVCppv':>8} {'HRmae':>7}")
    for snr in snrs:
        for kind in kinds:
            tp = fp = fn = 0
            hr_abs = []
            for rec in recs:
                path = os.path.join(args.data_dir, rec)
                try:
                    sig, fields = wfdb.rdsamp(path)
                    ann = wfdb.rdann(path, "atr")
                except Exception:
                    continue
                fs = float(fields["fs"])
                x = sig[:args.limit_samples, 0] if args.limit_samples else sig[:, 0]
                ref_idx = np.array([ann.sample[i] for i in range(len(ann.sample))
                                    if ann.symbol[i] in PVC_SYMS
                                    and (not args.limit_samples or ann.sample[i] < len(x))])
                xn = _add_noise(x, fs, kind, snr, rng, real=real.get(kind))
                a = eca.analyze(xn, fs)
                if not a.get("ok"):
                    continue
                det_pvc = np.array([b["idx"] for b in a["beats"] if b["type"] == "PVC"])
                tol = int(0.15 * fs)
                pairs = _match(ref_idx, det_pvc, tol) if len(ref_idx) else []
                tp += len(pairs)
                fn += len(ref_idx) - len(pairs)
                fp += len(det_pvc) - len(pairs)
                hr_ref = 60.0 / np.median(np.diff(ann.sample) / fs)
                hr_abs.append(abs(hr_ref - a["heart_rate"]))
            se = tp / (tp + fn) if tp + fn else float("nan")
            ppv = tp / (tp + fp) if tp + fp else float("nan")
            mae = np.mean(hr_abs) if hr_abs else float("nan")
            print(f"{snr:>5} {kind:>6} {se:>7.2f} {ppv:>8.2f} {mae:>7.1f}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name, fn in (("beats", cmd_beats), ("af", cmd_af), ("noise", cmd_noise)):
        sp = sub.add_parser(name)
        sp.add_argument("--data-dir", required=True)
        sp.add_argument("--limit-samples", type=int, default=0,
                        help="cap samples per record (speed); 0 = full record")
        if name == "noise":
            sp.add_argument("--noise-dir", default="",
                            help="dir with nstdb em/ma/bw noise records (real noise)")
        if name == "af":
            sp.add_argument("--ckpt", default="",
                            help="RhythmCNN checkpoint to score (default: models/rhythm_cnn.pt)")
        sp.set_defaults(func=fn)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
