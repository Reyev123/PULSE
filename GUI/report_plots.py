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
