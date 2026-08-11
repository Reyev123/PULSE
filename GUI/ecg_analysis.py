"""Single-lead beat analysis: R-peak detection and PAC/PVC estimation.

These are automated heuristic estimates from ONE lead. Distinguishing PACs
from PVCs without a P-wave-clear lead is inherently uncertain; treat the
counts as a screening aid, not a diagnosis.

Method:
  * band-pass 5-15 Hz to emphasise the QRS, then a Pan-Tompkins-style
    derivative/square/integrate to find R-peaks;
  * a beat is "premature" if the RR interval preceding it is much shorter
    than the local median RR;
  * a premature beat with a WIDE QRS is counted as a PVC, otherwise a PAC.
"""

import warnings

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

_PREMATURITY_RATIO = 0.85   # RR < 85% of the sinus-cycle reference => premature
_WIDE_QRS_S = 0.12          # >=120 ms QRS => wide (ventricular)
_MORPH_CORR_THRESH = 0.85   # QRS correlation below this vs sinus template => aberrant


def _bandpass(sig, fs, lo=5.0, hi=15.0, order=2):
    nyq = 0.5 * fs
    low = max(lo, 0.5) / nyq
    high = min(hi, nyq * 0.99) / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, sig)


def _detect_rpeaks(sig, fs):
    """Return (rpeak_indices, morphology_signal). Detection uses a 5-15 Hz
    band; morphology_signal is a wider 0.5-40 Hz band for QRS-width measies."""
    f = _bandpass(sig, fs)
    deriv = np.gradient(f)
    squared = deriv * deriv
    win = max(1, int(0.15 * fs))
    integrated = np.convolve(squared, np.ones(win) / win, mode="same")

    distance = max(1, int(0.30 * fs))  # refractory: <=200 bpm
    height = np.mean(integrated) + 0.5 * np.std(integrated)
    peaks, _ = find_peaks(integrated, distance=distance, height=height)

    morph = _bandpass(sig, fs, lo=0.5, hi=40.0)
    # refine each peak to the local |QRS| maximum within +/-50 ms
    r = max(1, int(0.05 * fs))
    refined = []
    for p in peaks:
        a, b = max(0, p - r), min(len(morph), p + r)
        refined.append(a + int(np.argmax(np.abs(morph[a:b]))))
    return np.array(sorted(set(refined))), morph


def _qrs_width(f, fs, peak):
    """Width (s) where |signal| stays above 50% of the R-peak amplitude."""
    amp = abs(f[peak])
    if amp == 0:
        return 0.0
    thr = 0.5 * amp
    i = peak
    while i > 0 and abs(f[i]) > thr:
        i -= 1
    j = peak
    while j < len(f) - 1 and abs(f[j]) > thr:
        j += 1
    return (j - i) / fs


def _classify(rpeaks, widths, p_present, fs, sig):
    """Return (pac, pvc, beats) given per-beat features.

    Premature beats are split by QRS morphology: a beat that is wide OR whose
    QRS correlates poorly with the sinus template is a PVC; otherwise a PAC.
    beats: list of {idx, t, type} where type in {normal, PAC, PVC}.
    """
    n = len(rpeaks)
    rr = np.diff(rpeaks) / fs
    valid_w = widths[np.isfinite(widths)]
    med_w = float(np.median(valid_w)) if len(valid_w) else _WIDE_QRS_S

    beats = [{"idx": int(rpeaks[i]), "t": round(float(rpeaks[i]) / fs, 3), "type": "normal"}
             for i in range(n)]

    # Pass 1: prematurity via an upper-percentile sinus-cycle reference.
    premature = np.zeros(n, dtype=bool)
    for k in range(1, n):
        lo, hi = max(0, k - 1 - 8), min(len(rr), k - 1 + 9)
        ref = float(np.percentile(rr[lo:hi], 60))
        if rr[k - 1] < _PREMATURITY_RATIO * ref:
            premature[k] = True

    # QRS windows and a sinus template from the non-premature beats.
    hw = max(1, int(0.12 * fs))
    windows = {}
    for i in range(n):
        a, b = int(rpeaks[i]) - hw, int(rpeaks[i]) + hw
        if a >= 0 and b <= len(sig):
            windows[i] = sig[a:b]
    normal_idx = [i for i in range(n) if not premature[i] and i in windows]
    template = (np.median(np.stack([windows[i] for i in normal_idx]), axis=0)
                if len(normal_idx) >= 3 else None)

    pac = pvc = 0
    for k in range(1, n):
        if not premature[k]:
            continue
        w = widths[k]
        wide = (np.isfinite(w) and (w >= _WIDE_QRS_S or w > 1.3 * med_w))
        aberrant = False
        if template is not None and k in windows:
            wk = windows[k]
            if np.std(wk) > 0 and np.std(template) > 0:
                aberrant = float(np.corrcoef(wk, template)[0, 1]) < _MORPH_CORR_THRESH
        if wide or aberrant:
            beats[k]["type"] = "PVC"
            pvc += 1
        else:
            beats[k]["type"] = "PAC"
            pac += 1
    return pac, pvc, beats


def _analyze_nk(sig, fs):
    """neurokit2 path: cleaned R-peaks + delineated QRS width and P-wave."""
    import neurokit2 as nk

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cleaned = nk.ecg_clean(sig, sampling_rate=fs)
        _, info = nk.ecg_peaks(cleaned, sampling_rate=fs)
        rpeaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
        if len(rpeaks) < 3:
            return {"ok": False, "reason": "too few beats detected"}
        try:
            _, waves = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=fs, method="dwt")
            onsets = np.asarray(waves.get("ECG_R_Onsets", []), dtype=float)
            offsets = np.asarray(waves.get("ECG_R_Offsets", []), dtype=float)
            ppeaks = np.asarray(waves.get("ECG_P_Peaks", []), dtype=float)
        except Exception:
            onsets = offsets = ppeaks = np.full(len(rpeaks), np.nan)

    n = len(rpeaks)

    def _al(a):  # pad/truncate to n
        a = np.asarray(a, dtype=float)
        if len(a) < n:
            a = np.concatenate([a, np.full(n - len(a), np.nan)])
        return a[:n]

    onsets, offsets, ppeaks = _al(onsets), _al(offsets), _al(ppeaks)
    widths = (offsets - onsets) / fs
    p_present = np.array([
        np.isfinite(ppeaks[i]) and 0.06 <= (rpeaks[i] - ppeaks[i]) / fs <= 0.30
        for i in range(n)
    ])

    hr = 60.0 / float(np.median(np.diff(rpeaks) / fs))
    pac, pvc, beats = _classify(rpeaks, widths, p_present, fs, cleaned)
    return _result(n, hr, pac, pvc, beats, engine="neurokit2")


def _analyze_basic(sig, fs):
    """Fallback path using the local scipy detector (no P-wave info)."""
    peaks, f = _detect_rpeaks(np.asarray(sig, dtype=float), fs)
    n = len(peaks)
    if n < 3:
        return {"ok": False, "reason": "too few beats detected"}
    widths = np.array([_qrs_width(f, fs, p) for p in peaks])
    p_present = np.zeros(n, dtype=bool)   # unknown -> treat missing P as ventricular hint
    # Without P-wave info, rely on width alone: pretend P present so narrow=>PAC.
    p_present[:] = True
    hr = 60.0 / float(np.median(np.diff(peaks) / fs))
    pac, pvc, beats = _classify(peaks, widths, p_present, fs, f)
    return _result(n, hr, pac, pvc, beats, engine="basic")


def _result(n, hr, pac, pvc, beats, engine):
    return {
        "ok": True,
        "engine": engine,
        "total_beats": int(n),
        "heart_rate": round(hr, 1),
        "pac_count": int(pac),
        "pvc_count": int(pvc),
        "pac_pct": round(100.0 * pac / n, 1),
        "pvc_pct": round(100.0 * pvc / n, 1),
        "beats": beats,
    }


def analyze(sig_mv, fs):
    """Return beat metrics + per-beat annotations (or {'ok': False, ...}).

    Uses neurokit2 when available for P-wave-aware PAC/PVC discrimination,
    falling back to a scipy-only heuristic.
    """
    fs = float(fs)
    if sig_mv is None or len(sig_mv) < fs * 2:
        return {"ok": False, "reason": "signal too short for beat analysis"}
    sig = np.asarray(sig_mv, dtype=float)
    try:
        res = _analyze_nk(sig, fs)
        if res.get("ok"):
            return res
    except Exception:
        pass
    return _analyze_basic(sig, fs)


def format_text(a):
    """Human-readable block for the report / PDF."""
    if not a or not a.get("ok"):
        reason = (a or {}).get("reason", "not available")
        return f"Automated beat analysis: {reason}."
    return (
        "Automated beat analysis (single lead — screening estimate):\n"
        f"  Heart rate: {a['heart_rate']} bpm\n"
        f"  Total beats: {a['total_beats']}\n"
        f"  PACs: {a['pac_count']} ({a['pac_pct']}%)\n"
        f"  PVCs: {a['pvc_count']} ({a['pvc_pct']}%)"
    )


def compute_trends(a, duration_s=None):
    """Time-course summaries for the report/GUI: HR trend, per-minute ectopy,
    HR min/avg/max, longest pause. Returns {'ok': False} if no beats."""
    if not a or not a.get("ok") or not a.get("beats"):
        return {"ok": False}
    ts = np.array([b["t"] for b in a["beats"]], dtype=float)
    types = [b["type"] for b in a["beats"]]
    if duration_s is None:
        duration_s = float(ts[-1]) if len(ts) else 0.0
    rr = np.diff(ts)
    hr_t = ts[1:]
    hr_bpm = 60.0 / np.clip(rr, 1e-3, None)
    good = (rr >= 0.3) & (rr <= 2.0)  # drop artifact RR before HR stats
    hb = hr_bpm[good]
    hr_min = float(np.percentile(hb, 2)) if len(hb) else 0.0
    hr_avg = float(np.median(hb)) if len(hb) else 0.0
    hr_max = float(np.percentile(hb, 98)) if len(hb) else 0.0
    longest_pause = float(np.max(rr)) if len(rr) else 0.0

    nmin = max(1, int(np.ceil(duration_s / 60.0)))
    per_min = [{"minute": m, "pac": 0, "pvc": 0, "beats": 0} for m in range(nmin)]
    for t, typ in zip(ts, types):
        m = min(nmin - 1, int(t // 60))
        per_min[m]["beats"] += 1
        if typ == "PVC":
            per_min[m]["pvc"] += 1
        elif typ == "PAC":
            per_min[m]["pac"] += 1

    step = max(1, len(hr_t) // 1500)  # cap stored/plotted points
    hr_series = [[round(float(hr_t[i]), 2), round(float(hr_bpm[i]), 1)]
                 for i in range(0, len(hr_t), step)]
    rr_series = [round(float(x), 3) for x in rr]
    return {
        "ok": True,
        "duration_s": round(duration_s, 1),
        "hr_min": round(hr_min), "hr_avg": round(hr_avg), "hr_max": round(hr_max),
        "longest_pause_s": round(longest_pause, 2),
        "per_minute": per_min,
        "hr_series": hr_series,
        "rr_series": rr_series,
    }


def _densest_center(event_t, window):
    """Center (s) of the `window`-second span containing the most events."""
    event_t = np.sort(np.asarray(event_t, dtype=float))
    n = len(event_t)
    best_i, best_n, j = 0, 0, 0
    for i in range(n):
        while j < n and event_t[j] <= event_t[i] + window:
            j += 1
        if j - i > best_n:
            best_n, best_i = j - i, i
    return float(event_t[best_i] + window / 2.0)


def select_report_strips(a, window=10.0):
    """Representative windows (seconds) for the report: a clean sinus strip plus
    the densest PVC and PAC clusters. Returns [{label, kind, start}]."""
    if not a or not a.get("ok") or not a.get("beats"):
        return []
    ts = np.array([b["t"] for b in a["beats"]], dtype=float)
    types = np.array([b["type"] for b in a["beats"]])
    dur = float(ts[-1]) if len(ts) else 0.0
    half = window / 2.0

    def clamp(s):
        return round(max(0.0, min(s, max(0.0, dur - window))), 2)

    pvc_t = ts[types == "PVC"]
    pac_t = ts[types == "PAC"]
    ect_t = np.sort(np.concatenate([pvc_t, pac_t])) if len(pvc_t) + len(pac_t) else np.array([])

    strips = []
    best = None
    for s in np.arange(0, max(1.0, dur - window), 5.0):
        if not len(ect_t) or not np.any((ect_t >= s) & (ect_t <= s + window)):
            d = abs((s + half) - dur / 2)
            if best is None or d < best[0]:
                best = (d, s)
    if best is not None:
        strips.append({"label": "Normal sinus", "kind": "normal", "start": clamp(best[1])})
    if len(pvc_t):
        strips.append({"label": "Frequent PVCs", "kind": "PVC",
                       "start": clamp(_densest_center(pvc_t, window) - half)})
    if len(pac_t):
        strips.append({"label": "Atrial ectopy (PAC)", "kind": "PAC",
                       "start": clamp(_densest_center(pac_t, window) - half)})
    return strips


def next_event_time(a, after_t, kind):
    """First beat time of `kind` ('PVC'/'PAC') strictly after after_t, wrapping
    to the earliest such beat. Returns None if none exist."""
    if not a or not a.get("ok") or not a.get("beats"):
        return None
    evs = [b["t"] for b in a["beats"] if b["type"] == kind]
    if not evs:
        return None
    later = [t for t in evs if t > after_t + 0.5]
    return later[0] if later else evs[0]
