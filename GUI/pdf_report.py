"""Build a PDF report (input ECG image + generated findings) with ReportLab."""

import io
import textwrap

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

import signal_io as sio
import ecg_analysis as eca
import report_plots as rp

_MARGIN = 0.8 * inch
_DISCLAIMER = ("Research use only. Not an FDA-cleared diagnostic. "
               "Single-lead output may be unreliable for 12-lead-only findings.")


def _draw_record(c, width, height, title, image_bytes, report_text, meta):
    """Render one record onto the current page (caller handles showPage)."""
    y = height - _MARGIN

    c.setFont("Helvetica-Bold", 16)
    c.drawString(_MARGIN, y, title)
    y -= 0.35 * inch

    c.setFont("Helvetica", 9)
    for k, v in (meta or {}).items():
        c.drawString(_MARGIN, y, f"{k}: {v}")
        y -= 0.18 * inch
    y -= 0.05 * inch

    if image_bytes:
        img = ImageReader(io.BytesIO(image_bytes))
        iw, ih = img.getSize()
        draw_w = width - 2 * _MARGIN
        draw_h = min(ih * (draw_w / iw), 3.2 * inch)
        c.drawImage(img, _MARGIN, y - draw_h, width=draw_w, height=draw_h,
                    preserveAspectRatio=True, anchor="n")
        y -= draw_h + 0.3 * inch

    c.setFont("Helvetica-Bold", 12)
    c.drawString(_MARGIN, y, "Findings")
    y -= 0.25 * inch

    c.setFont("Helvetica", 10)
    for paragraph in (report_text or "").split("\n"):
        for line in textwrap.wrap(paragraph, width=95) or [""]:
            if y < _MARGIN:
                c.showPage()
                y = height - _MARGIN
                c.setFont("Helvetica", 10)
            c.drawString(_MARGIN, y, line)
            y -= 0.2 * inch

    c.setFont("Helvetica-Oblique", 7)
    c.drawString(_MARGIN, _MARGIN * 0.6, _DISCLAIMER)


def build_pdf(image_bytes, report_text, meta=None):
    """Return single-record PDF bytes."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    _draw_record(c, width, height, "PULSE Single-Lead ECG Report",
                 image_bytes, report_text, meta)
    c.showPage()
    c.save()
    return buf.getvalue()


def build_batch_pdf(records):
    """Return multi-record PDF bytes.

    records: list of dicts with keys id, image_bytes, report, meta.
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    for rec in records:
        _draw_record(c, width, height,
                     f"PULSE ECG Report — {rec.get('id', '')}",
                     rec.get("image_bytes"), rec.get("report"), rec.get("meta"))
        c.showPage()
    c.save()
    return buf.getvalue()


# --- full multi-section report ---------------------------------------------
def _footer(c):
    c.setFont("Helvetica-Oblique", 7)
    c.drawString(_MARGIN, _MARGIN * 0.6, _DISCLAIMER)


def _fmt_duration(sec):
    sec = int(sec or 0)
    return f"{sec // 60} min {sec % 60:02d} s"


def _draw_image(c, png, x, y_top, max_w, max_h):
    """Draw a PNG with its top at y_top; return the y below it."""
    img = ImageReader(io.BytesIO(png))
    iw, ih = img.getSize()
    w = max_w
    h = ih * (w / iw)
    if h > max_h:
        h = max_h
        w = iw * (h / ih)
    c.drawImage(img, x, y_top - h, width=w, height=h, preserveAspectRatio=True, anchor="n")
    return y_top - h


def _summary_page(c, width, height, analysis, trends, report_text, meta):
    y = height - _MARGIN
    c.setFont("Helvetica-Bold", 16)
    c.drawString(_MARGIN, y, "Single-Lead ECG Report")
    y -= 0.32 * inch
    c.setFont("Helvetica", 9)
    for k, v in (meta or {}).items():
        c.drawString(_MARGIN, y, f"{k}: {v}")
        y -= 0.16 * inch
    y -= 0.08 * inch

    rows = [
        ("Duration", _fmt_duration(trends.get("duration_s"))),
        ("Heart rate (min / avg / max)",
         f"{trends.get('hr_min','-')} / {trends.get('hr_avg','-')} / {trends.get('hr_max','-')} bpm"),
        ("Total beats", str(analysis.get("total_beats", "-"))),
        ("PVC burden", f"{analysis.get('pvc_count','-')} ({analysis.get('pvc_pct','-')}%)"),
        ("PAC burden", f"{analysis.get('pac_count','-')} ({analysis.get('pac_pct','-')}%)"),
        ("Rhythm (RhythmCNN)", analysis.get("rhythm") or "not classified"),
        ("Longest RR pause", f"{trends.get('longest_pause_s','-')} s"),
    ]
    c.setFont("Helvetica-Bold", 12)
    c.drawString(_MARGIN, y, "Summary")
    y -= 0.24 * inch
    c.setFont("Helvetica", 10)
    for label, val in rows:
        c.drawString(_MARGIN + 6, y, label)
        c.drawString(_MARGIN + 3.1 * inch, y, str(val))
        y -= 0.2 * inch
    y -= 0.06 * inch

    c.setFont("Helvetica-Bold", 12)
    c.drawString(_MARGIN, y, "Findings")
    y -= 0.22 * inch
    c.setFont("Helvetica", 10)
    for paragraph in (report_text or "").split("\n"):
        for line in textwrap.wrap(paragraph, width=95) or [""]:
            if y < _MARGIN + 2.2 * inch:
                break
            c.drawString(_MARGIN, y, line)
            y -= 0.19 * inch

    if trends.get("ok"):
        y -= 0.05 * inch
        c.setFont("Helvetica-Bold", 11)
        c.drawString(_MARGIN, y, "Heart-rate trend")
        y -= 0.12 * inch
        _draw_image(c, rp.hr_tachogram_png(trends), _MARGIN, y,
                    width - 2 * _MARGIN, 1.5 * inch)
    _footer(c)


def _strips_pages(c, width, height, mv, fs, strips, window):
    per_page = 3
    for i, strip in enumerate(strips):
        if i % per_page == 0:
            if i > 0:
                c.showPage()
            y = height - _MARGIN
            c.setFont("Helvetica-Bold", 14)
            c.drawString(_MARGIN, y, "Representative strips")
            y -= 0.34 * inch
        c.setFont("Helvetica-Bold", 10)
        c.drawString(_MARGIN, y, f"{strip['label']} — at {_fmt_duration(strip['start'])}")
        y -= 0.14 * inch
        png = sio.render_ecg_png(mv, fs, seconds=window, start=strip["start"])
        y = _draw_image(c, png, _MARGIN, y, width - 2 * _MARGIN, 1.9 * inch)
        y -= 0.3 * inch
    _footer(c)


def _trends_page(c, width, height, trends):
    y = height - _MARGIN
    c.setFont("Helvetica-Bold", 14)
    c.drawString(_MARGIN, y, "Trends")
    y -= 0.34 * inch
    c.setFont("Helvetica-Bold", 11)
    c.drawString(_MARGIN, y, "Ectopy per minute")
    y -= 0.12 * inch
    y = _draw_image(c, rp.ectopy_bar_png(trends), _MARGIN, y, width - 2 * _MARGIN, 1.9 * inch)
    y -= 0.35 * inch
    c.setFont("Helvetica-Bold", 11)
    c.drawString(_MARGIN, y, "Poincaré (RR) plot")
    y -= 0.12 * inch
    _draw_image(c, rp.poincare_png(trends), _MARGIN, y, 2.6 * inch, 2.6 * inch)
    _footer(c)


def _disclosure_pages(c, width, height, mv, fs):
    """Append the whole recording as 60 s/line 'full disclosure' pages."""
    pages = rp.full_disclosure_pngs(mv, fs)
    for i, png in enumerate(pages):
        y = height - _MARGIN * 0.7
        c.setFont("Helvetica-Bold", 12)
        title = "Full disclosure — 60 s / line"
        c.drawString(_MARGIN, y, title if i == 0 else f"{title} (cont. {i + 1})")
        _draw_image(c, png, _MARGIN, y - 0.18 * inch,
                    width - 2 * _MARGIN, height - 1.7 * inch)
        _footer(c)
        c.showPage()


def build_full_pdf(mv, fs, analysis, report_text, meta=None, window=10.0, trends=None,
                   include_full_disclosure=True):
    """Multi-section report: summary + representative strips + trends + full disclosure.

    mv: full-resolution millivolt signal (list/array); fs: sampling rate.
    """
    import numpy as np
    mv = np.asarray(mv, dtype=float)
    if trends is None:
        trends = eca.compute_trends(analysis, duration_s=len(mv) / fs)
    strips = eca.select_report_strips(analysis, window=window) if analysis.get("ok") else []

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    _summary_page(c, width, height, analysis, trends, report_text, meta)
    c.showPage()
    if strips:
        _strips_pages(c, width, height, mv, fs, strips, window)
        c.showPage()
    if trends.get("ok"):
        _trends_page(c, width, height, trends)
        c.showPage()
    if include_full_disclosure:
        _disclosure_pages(c, width, height, mv, fs)
    c.save()
    return buf.getvalue()

