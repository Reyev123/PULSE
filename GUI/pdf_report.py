"""Build a PDF report (input ECG image + generated findings) with ReportLab."""

import io
import textwrap

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

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
