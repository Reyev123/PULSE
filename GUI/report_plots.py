"""Matplotlib trend/scatter plots for the full PDF report (returns PNG bytes)."""

import io

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PVC = "#b22222"
_PAC = "#b8860b"
_BLUE = "#1a6ebd"


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def hr_tachogram_png(trends, size=(6.6, 1.7)):
    """Heart-rate over time with min/avg/max reference lines."""
    fig, ax = plt.subplots(figsize=size)
    series = trends.get("hr_series") or []
    if series:
        t = [p[0] / 60.0 for p in series]  # minutes
        hr = [p[1] for p in series]
        ax.plot(t, hr, color=_BLUE, linewidth=0.8)
    for key, style in (("hr_min", ":"), ("hr_avg", "--"), ("hr_max", ":")):
        v = trends.get(key)
        if v:
            ax.axhline(v, color="#888", linestyle=style, linewidth=0.7)
    ax.set_ylabel("HR (bpm)", fontsize=8)
    ax.set_xlabel("Time (min)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, color="#eee", linewidth=0.5)
    return _png(fig)


def ectopy_bar_png(trends, size=(6.6, 1.7)):
    """Stacked per-minute PVC (red) and PAC (gold) counts."""
    fig, ax = plt.subplots(figsize=size)
    pm = trends.get("per_minute") or []
    if pm:
        mins = [d["minute"] for d in pm]
        pvc = [d["pvc"] for d in pm]
        pac = [d["pac"] for d in pm]
        ax.bar(mins, pvc, color=_PVC, label="PVC")
        ax.bar(mins, pac, bottom=pvc, color=_PAC, label="PAC")
        ax.legend(fontsize=7, loc="upper right")
    ax.set_ylabel("Beats / min", fontsize=8)
    ax.set_xlabel("Time (min)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, axis="y", color="#eee", linewidth=0.5)
    return _png(fig)


def poincare_png(trends, size=(2.4, 2.4)):
    """RR(n) vs RR(n+1) scatter — rhythm-variability fingerprint."""
    fig, ax = plt.subplots(figsize=size)
    rr = trends.get("rr_series") or []
    if len(rr) > 2:
        rr = np.asarray(rr)
        ax.scatter(rr[:-1], rr[1:], s=4, color=_BLUE, alpha=0.4, edgecolors="none")
        lim = float(np.percentile(rr, 98)) if len(rr) else 1.5
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
    ax.set_xlabel("RRₙ (s)", fontsize=8)
    ax.set_ylabel("RRₙ₊₁ (s)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal", "box")
    return _png(fig)


def full_disclosure_pngs(mv, fs, seconds_per_row=60.0, rows_per_page=8, size=(7.2, 9.0)):
    """Render the ENTIRE recording as stacked rows (~60 s/line); one PNG per page."""
    mv = np.asarray(mv, dtype=float)
    n = len(mv)
    dur = n / fs if fs else 0
    row_samples = max(1, int(seconds_per_row * fs))
    total_rows = max(1, int(np.ceil(n / row_samples)))
    dec = max(1, row_samples // 2000)  # decimate to ~2000 pts/row
    amp = float(np.percentile(np.abs(mv), 99)) or 1.0
    pages = []
    for p0 in range(0, total_rows, rows_per_page):
        rows = list(range(p0, min(p0 + rows_per_page, total_rows)))
        fig, ax = plt.subplots(figsize=size)
        for ri, r in enumerate(rows):
            a = r * row_samples
            b = min(n, a + row_samples)
            seg = mv[a:b][::dec]
            x = np.arange(len(seg)) * dec / fs
            baseline = -ri
            ax.plot(x, baseline + seg / (3.0 * amp), color="#111", linewidth=0.4)
            secs = int(r * seconds_per_row)
            ax.text(-1.5, baseline, f"{secs // 60:02d}:{secs % 60:02d}",
                    fontsize=6, ha="right", va="center", color="#666")
        for xs in range(0, int(seconds_per_row) + 1, 5):
            ax.axvline(xs, color="#f0d6d6", linewidth=0.3, zorder=0)
        ax.set_xlim(-3, seconds_per_row)
        ax.set_ylim(-len(rows), 1)
        ax.set_xticks(range(0, int(seconds_per_row) + 1, 10))
        ax.tick_params(labelsize=6)
        ax.set_yticks([])
        ax.set_xlabel(f"seconds within row ({int(seconds_per_row)} s/row) · {dur:.0f}s total",
                      fontsize=7)
        for s in ax.spines.values():
            s.set_visible(False)
        pages.append(_png(fig))
    return pages

