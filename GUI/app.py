"""Single-lead ECG — Dash/Plotly GUI.

Upload raw single-lead signals (Frontier X Plus csv/txt/npy) or pre-rendered
ECG images, view the trace, run analysis (signal facts + RhythmCNN), read the
Ollama-generated report, and export input + report as a PDF.

Run (inside the pulse-llava env):
    python GUI/app.py
then open http://127.0.0.1:8050
"""

import base64
import datetime as dt
import os
import sys

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output, State, no_update, ctx

import signal_io as sio
import ecg_analysis as eca
import ecg_digitize as edg
import narrative
from pdf_report import build_pdf, build_batch_pdf, build_full_pdf

# RhythmCNN (optional): predicts N/AF/Other/Noisy if a trained checkpoint exists.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
try:
    import rhythm_infer
except Exception:
    rhythm_infer = None

DEFAULT_PROMPT = "Please write a clinical report based on this single-lead ECG image."

app = Dash(__name__, title="PULSE Single-Lead ECG", suppress_callback_exceptions=True)
# Allow large single-lead recordings / images to upload without a 413.
app.server.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024


def _decode_upload(contents):
    """dcc.Upload contents 'data:<mime>;base64,<payload>' -> raw bytes."""
    _header, payload = contents.split(",", 1)
    return base64.b64decode(payload)


def _b64_png(png_bytes):
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


def _stat_cards(analysis):
    """Small metric cards for the GUI from an analysis dict."""
    if not analysis or not analysis.get("ok"):
        reason = (analysis or {}).get("reason", "raw signal required")
        return [html.Div(f"Beat analysis: {reason}", style={"color": "#888"})]

    def card(label, value, color="#333"):
        return html.Div(
            style={"padding": "8px 14px", "border": "1px solid #e0e0e0",
                   "borderRadius": "6px", "minWidth": "90px"},
            children=[html.Div(label, style={"fontSize": "12px", "color": "#777"}),
                      html.Div(value, style={"fontSize": "18px", "fontWeight": "bold",
                                             "color": color})],
        )

    cards = [
        card("Heart rate", f"{analysis['heart_rate']} bpm"),
        card("Total beats", str(analysis["total_beats"])),
        card("PACs", f"{analysis['pac_count']} ({analysis['pac_pct']}%)", "#b8860b"),
        card("PVCs", f"{analysis['pvc_count']} ({analysis['pvc_pct']}%)", "#b22222"),
    ]
    if analysis.get("rhythm"):
        cards.insert(0, card("Rhythm", analysis["rhythm"], "#1a6ebd"))
    return cards


def _apply_ecg_grid(fig):
    """Give a Plotly figure ECG-paper styling with readable time labels.

    Major ticks/labels are auto-spaced (readable at any duration); the fine
    0.2 s / 0.1 mV ECG grid is drawn as unlabeled minor gridlines.
    """
    fig.update_layout(plot_bgcolor="white", height=320, margin={"t": 40})
    fig.update_xaxes(gridcolor="#e6a3a3", zeroline=False, ticks="outside",
                     minor={"dtick": 0.2, "gridcolor": "#f2cccc", "showgrid": True})
    fig.update_yaxes(dtick=0.5, gridcolor="#e6a3a3", zeroline=False,
                     minor={"dtick": 0.1, "gridcolor": "#f2cccc", "showgrid": True})
    return fig


def _slice_window(n, fs, start, seconds):
    """Index bounds for the [start, start+seconds] window into an n-sample signal."""
    i0 = max(0, min(int(round((start or 0) * fs)), max(0, n - 1)))
    i1 = n if not seconds else min(n, i0 + int(round(seconds * fs)))
    return i0, i1


def _ecg_figure(mv, fs, analysis, title, start=0.0, seconds=None):
    """Windowed line trace + PAC/PVC markers on ECG-paper grid (WebGL)."""
    i0, i1 = _slice_window(len(mv), fs, start, seconds)
    wt = np.arange(i0, i1) / fs
    fig = go.Figure(go.Scattergl(x=wt, y=mv[i0:i1], mode="lines",
                                 line={"width": 1, "color": "#111"}, name="ECG"))
    if analysis and analysis.get("ok") and analysis.get("beats"):
        for typ, color, symbol in (("PAC", "#b8860b", "diamond"), ("PVC", "#b22222", "x")):
            pts = [(b["t"], float(mv[b["idx"]])) for b in analysis["beats"]
                   if b["type"] == typ and i0 <= b["idx"] < i1]
            if pts:
                xs, ys = zip(*pts)
                fig.add_scatter(x=list(xs), y=list(ys), mode="markers", name=typ,
                                marker={"color": color, "symbol": symbol, "size": 11})
    fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="mV")
    fig = _apply_ecg_grid(fig)
    seg = mv[i0:i1]
    span = (i1 - i0) / fs if fs else 0
    if len(seg):
        lo, hi = float(np.min(seg)), float(np.max(seg))
        pad = max(0.3, 0.15 * (hi - lo))
        # Lock amplitude so zoom/pan can't squash the trace; x stays zoomable.
        fig.update_yaxes(range=[lo - pad, hi + pad], fixedrange=True)
        fig.update_xaxes(range=[i0 / fs, max(i0 + 1, i1 - 1) / fs])
    if span > 30:  # too many minor lines get sluggish on wide views
        fig.update_xaxes(minor={"showgrid": False})
    return fig


def _overview_figure(mv, fs, analysis, start, window):
    """Whole-recording min/max envelope with ectopy ticks and a window highlight."""
    n = len(mv)
    dur = n / fs if fs else 0
    nb = min(1500, max(1, n // 10))
    edges = np.linspace(0, n, nb + 1).astype(int)
    xs, ymin, ymax = [], [], []
    for i in range(nb):
        a, b = edges[i], edges[i + 1]
        if b <= a:
            continue
        seg = mv[a:b]
        xs.append((a + b) / 2 / fs)
        ymin.append(float(seg.min()))
        ymax.append(float(seg.max()))
    fig = go.Figure()
    if xs:
        fig.add_trace(go.Scatter(x=xs + xs[::-1], y=ymax + ymin[::-1], fill="toself",
                                 fillcolor="rgba(122,169,214,0.35)", line={"width": 0},
                                 hoverinfo="skip", showlegend=False))
        ytop = max(ymax)
        if analysis and analysis.get("ok") and analysis.get("beats"):
            for typ, color in (("PVC", "#b22222"), ("PAC", "#b8860b")):
                te = [b["t"] for b in analysis["beats"] if b["type"] == typ]
                if te:
                    fig.add_trace(go.Scattergl(
                        x=te, y=[ytop] * len(te), mode="markers", name=typ,
                        marker={"color": color, "symbol": "line-ns-open", "size": 9},
                        hovertemplate=typ + " %{x:.1f}s<extra></extra>"))
        fig.add_vrect(x0=start, x1=min(dur, start + window), fillcolor="#1a6ebd",
                      opacity=0.15, line_width=1, line_color="#1a6ebd")
    fig.update_layout(height=150, margin={"t": 10, "b": 26, "l": 44, "r": 10},
                      plot_bgcolor="white", showlegend=False, clickmode="event",
                      xaxis_title="Time (s) — click to jump")
    fig.update_yaxes(fixedrange=True, showticklabels=False)
    fig.update_xaxes(range=[0, dur], gridcolor="#eee")
    return fig


def _hr_trend_figure(trends):
    """HR-over-time tachogram with min/avg/max reference lines."""
    fig = go.Figure()
    if trends and trends.get("hr_series"):
        xs = [p[0] for p in trends["hr_series"]]
        ys = [p[1] for p in trends["hr_series"]]
        fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines",
                                   line={"color": "#1a6ebd", "width": 1}))
        for key, dash in (("hr_min", "dot"), ("hr_avg", "dash"), ("hr_max", "dot")):
            if trends.get(key):
                fig.add_hline(y=trends[key], line={"color": "#999", "dash": dash, "width": 1})
    fig.update_layout(height=170, margin={"t": 12, "b": 32, "l": 46, "r": 10},
                      plot_bgcolor="white", showlegend=False,
                      xaxis_title="Time (s)", yaxis_title="HR (bpm)")
    fig.update_xaxes(gridcolor="#eee")
    fig.update_yaxes(gridcolor="#eee")
    return fig


def _ectopy_trend_figure(trends):
    """Stacked per-minute PVC/PAC burden bars."""
    fig = go.Figure()
    if trends and trends.get("per_minute"):
        pm = trends["per_minute"]
        mins = [d["minute"] for d in pm]
        fig.add_trace(go.Bar(x=mins, y=[d["pvc"] for d in pm], name="PVC",
                             marker_color="#b22222"))
        fig.add_trace(go.Bar(x=mins, y=[d["pac"] for d in pm], name="PAC",
                             marker_color="#b8860b"))
    fig.update_layout(height=170, margin={"t": 12, "b": 32, "l": 46, "r": 10},
                      plot_bgcolor="white", barmode="stack",
                      xaxis_title="Time (min)", yaxis_title="Beats / min",
                      legend={"orientation": "h", "y": 1.15})
    fig.update_xaxes(gridcolor="#eee")
    fig.update_yaxes(gridcolor="#eee")
    return fig


def _add_rhythm(analysis, mv, fs):
    """Attach RhythmCNN prediction to the analysis dict when a model exists."""
    if rhythm_infer is None or not analysis or not analysis.get("ok"):
        return
    try:
        rh = rhythm_infer.predict(mv, fs)
        if rh:
            analysis["rhythm"] = f"{rh['name']} ({rh['prob'] * 100:.0f}%)"
    except Exception:
        pass


def _facts_from_analysis(analysis):
    """Structured facts for the narrative LLM."""
    if not analysis or not analysis.get("ok"):
        return {}
    return {
        "heart_rate": analysis["heart_rate"],
        "total_beats": analysis["total_beats"],
        "pac": f"{analysis['pac_count']} ({analysis['pac_pct']}%)",
        "pvc": f"{analysis['pvc_count']} ({analysis['pvc_pct']}%)",
        "rhythm": analysis.get("rhythm"),
    }



app.layout = html.Div(
    style={"maxWidth": "980px", "margin": "0 auto", "fontFamily": "Arial, sans-serif"},
    children=[
        html.H2("PULSE — Single-Lead ECG Analyzer"),
        html.P("Upload raw signal (.csv/.txt/.npy) or an ECG image (.png/.jpg). "
               "Research use only — not an FDA-cleared diagnostic."),

        # Shared raw-signal rendering parameters (used by both tabs).
        html.Div(
            style={"display": "flex", "gap": "16px", "alignItems": "center",
                   "flexWrap": "wrap", "marginBottom": "12px"},
            children=[
                html.Label("Raw sampling rate (Hz):"),
                dcc.Input(id="fs", type="number", value=500, min=1, style={"width": "90px"}),
                html.Label("Column:"),
                dcc.Input(id="column", type="number", value=0, min=0, style={"width": "70px"}),
                html.Label("Unit:"),
                dcc.Dropdown(id="unit", options=["mv", "uv", "raw"], value="mv",
                             clearable=False, style={"width": "110px"}),
                html.Label("Window (s):"),
                dcc.Input(id="seconds", type="number", value=10, min=1, style={"width": "80px"}),
            ],
        ),

        dcc.Tabs(id="tabs", value="single", children=[
            dcc.Tab(label="Single", value="single", children=[
                dcc.Upload(
                    id="upload",
                    children=html.Div(["Drag & drop or ", html.A("select a file")]),
                    style={"width": "100%", "height": "70px", "lineHeight": "70px",
                           "borderWidth": "1px", "borderStyle": "dashed",
                           "borderRadius": "6px", "textAlign": "center",
                           "margin": "12px 0"},
                    multiple=False,
                ),
                html.Div(id="status", style={"color": "#555", "marginBottom": "8px"}),
                html.H4("Overview \u2014 click to jump; ticks mark ectopy"),
                dcc.Graph(id="overview-graph", config={"displayModeBar": False}),
                html.Div(
                    style={"display": "flex", "gap": "10px", "alignItems": "center",
                           "flexWrap": "wrap", "margin": "4px 0 8px"},
                    children=[
                        html.Button("\u25c0 Prev", id="win-prev", n_clicks=0),
                        html.Button("Next \u25b6", id="win-next", n_clicks=0),
                        html.Button("\u2934 Next PVC", id="win-pvc", n_clicks=0),
                        html.Button("\u2934 Next PAC", id="win-pac", n_clicks=0),
                        html.Label("Window start (s):"),
                        dcc.Input(id="win-start", type="number", value=0, min=0,
                                  step=1, style={"width": "90px"}),
                    ],
                ),
                html.H4("Detail \u2014 selected window (drag to zoom, double-click to reset)"),
                dcc.Graph(id="ecg-graph"),
                html.Div(id="beat-stats", style={"display": "flex", "gap": "18px",
                         "flexWrap": "wrap", "margin": "6px 0 12px"}),
                html.H4("Trends"),
                dcc.Graph(id="hr-trend", config={"displayModeBar": False}),
                dcc.Graph(id="ectopy-trend", config={"displayModeBar": False}),
                html.H4("Rendered ECG image (model input)"),
                html.Img(id="ecg-image", style={"maxWidth": "100%", "border": "1px solid #ddd"}),
                html.H4("Prompt"),
                dcc.Textarea(id="prompt", value=DEFAULT_PROMPT,
                             style={"width": "100%", "height": "56px"}),
                html.Div(
                    style={"display": "flex", "gap": "12px", "margin": "12px 0"},
                    children=[
                        html.Button("Run Inference", id="run-btn", n_clicks=0),
                        html.Button("Export PDF", id="pdf-btn", n_clicks=0),
                    ],
                ),
                dcc.Loading(
                    type="default",
                    children=html.Div(
                        id="report",
                        style={"whiteSpace": "pre-wrap", "background": "#f7f7f7",
                               "padding": "12px", "borderRadius": "6px",
                               "minHeight": "60px"},
                    ),
                ),
                dcc.Download(id="download-pdf"),
                dcc.Store(id="store-png"),
                dcc.Store(id="store-report"),
                dcc.Store(id="store-meta"),
                dcc.Store(id="store-analysis"),
                dcc.Store(id="store-signal"),
                dcc.Store(id="store-trends"),
            ]),

            dcc.Tab(label="Batch", value="batch", children=[
                dcc.Upload(
                    id="batch-upload",
                    children=html.Div(["Drag & drop or ", html.A("select multiple files")]),
                    style={"width": "100%", "height": "70px", "lineHeight": "70px",
                           "borderWidth": "1px", "borderStyle": "dashed",
                           "borderRadius": "6px", "textAlign": "center",
                           "margin": "12px 0"},
                    multiple=True,
                ),
                html.H4("Prompt"),
                dcc.Textarea(id="batch-prompt", value=DEFAULT_PROMPT,
                             style={"width": "100%", "height": "56px"}),
                html.Div(
                    style={"display": "flex", "gap": "12px", "margin": "12px 0"},
                    children=[
                        html.Button("Run Batch Inference", id="run-batch-btn", n_clicks=0),
                        html.Button("Export JSONL", id="batch-jsonl-btn", n_clicks=0),
                        html.Button("Export Batch PDF", id="batch-pdf-btn", n_clicks=0),
                    ],
                ),
                html.Div(id="batch-status", style={"color": "#555", "marginBottom": "8px"}),
                dcc.Loading(
                    type="default",
                    children=dash_table.DataTable(
                        id="batch-table",
                        columns=[{"name": c, "id": c} for c in
                                 ("id", "type", "HR", "PAC %", "PVC %", "report", "status")],
                        data=[],
                        style_cell={"textAlign": "left", "whiteSpace": "normal",
                                    "height": "auto", "fontFamily": "Arial", "fontSize": "13px"},
                        style_cell_conditional=[{"if": {"column_id": "report"},
                                                 "maxWidth": "520px"}],
                        page_size=25,
                    ),
                ),
                dcc.Download(id="download-batch-jsonl"),
                dcc.Download(id="download-batch-pdf"),
                dcc.Store(id="store-batch"),
            ]),
        ]),
    ],
)


@app.callback(
    Output("ecg-graph", "figure"),
    Output("ecg-image", "src"),
    Output("status", "children"),
    Output("store-png", "data"),
    Output("store-meta", "data"),
    Output("beat-stats", "children"),
    Output("store-analysis", "data"),
    Output("report", "children", allow_duplicate=True),
    Output("store-report", "data", allow_duplicate=True),
    Output("store-signal", "data"),
    Output("win-start", "value", allow_duplicate=True),
    Output("overview-graph", "figure"),
    Output("hr-trend", "figure"),
    Output("ectopy-trend", "figure"),
    Output("store-trends", "data"),
    Input("upload", "contents"),
    State("upload", "filename"),
    Input("fs", "value"),
    Input("column", "value"),
    Input("unit", "value"),
    State("seconds", "value"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename, fs, column, unit, seconds):
    if not contents or not filename:
        return (no_update,) * 15

    data = _decode_upload(contents)
    meta = {"file": filename, "loaded": dt.datetime.now().isoformat(timespec="seconds")}
    analysis = None
    signal_store = None
    trends_store = None
    overview_fig = hr_fig = ectopy_fig = no_update
    window = float(seconds or 10)
    start0 = 0.0  # every new upload starts at the first window
    # Cleared on every new upload so stale results never linger.
    cleared_report, cleared_store = "", None

    try:
        if sio.is_image(filename):
            png = data
            secs = float(seconds or 10)
            dig = edg.digitize(sio.png_bytes_to_pil(data), seconds=secs)
            if dig.get("ok"):
                mv = np.asarray(dig["mv"], dtype=float)
                fs_v = dig["fs"]
                analysis = eca.analyze(mv, fs_v)
                _add_rhythm(analysis, mv, fs_v)
                fig = _ecg_figure(mv, fs_v, analysis,
                                  "Digitized image (experimental) — PAC/PVC marked",
                                  start=start0, seconds=window)
                signal_store = {"mv": [round(float(x), 3) for x in mv], "fs": fs_v,
                                "duration": round(len(mv) / fs_v, 2)}
                trends_store = eca.compute_trends(analysis, duration_s=len(mv) / fs_v)
                overview_fig = _overview_figure(mv, fs_v, analysis, start0, window)
                hr_fig = _hr_trend_figure(trends_store)
                ectopy_fig = _ectopy_trend_figure(trends_store)
                meta.update({"input_type": "image (digitized)",
                             "grid_detected": dig["grid_detected"],
                             "digitized_fs_hz": round(fs_v, 1)})
                if analysis.get("ok"):
                    meta.update({"heart_rate_bpm": analysis["heart_rate"],
                                 "pac": f"{analysis['pac_count']} ({analysis['pac_pct']}%)",
                                 "pvc": f"{analysis['pvc_count']} ({analysis['pvc_pct']}%)"})
                status = f"Loaded image: {filename} (digitized, grid={'yes' if dig['grid_detected'] else 'no'})"
            else:
                fig = _apply_ecg_grid(go.Figure())
                fig.update_layout(title=f"Image not digitized: {dig.get('reason', '')}")
                analysis = {"ok": False, "reason": dig.get("reason", "digitization failed")}
                meta.update({"input_type": "image"})
                status = f"Loaded image: {filename} (no beat markers — {dig.get('reason', '')})"
        elif sio.is_signal(filename):
            sig = sio.parse_signal_bytes(data, filename, column=int(column or 0))
            mv = sio.signal_to_mv(sig, unit=unit)
            fs_v = float(fs or 500)
            analysis = eca.analyze(mv, fs_v)
            _add_rhythm(analysis, mv, fs_v)
            png = sio.render_ecg_png(mv, fs_v, seconds=window, start=start0)
            fig = _ecg_figure(mv, fs_v, analysis, "Single-lead ECG — PAC/PVC marked",
                              start=start0, seconds=window)
            signal_store = {"mv": [round(float(x), 3) for x in mv], "fs": fs_v,
                            "duration": round(len(mv) / fs_v, 2)}
            trends_store = eca.compute_trends(analysis, duration_s=len(mv) / fs_v)
            overview_fig = _overview_figure(mv, fs_v, analysis, start0, window)
            hr_fig = _hr_trend_figure(trends_store)
            ectopy_fig = _ectopy_trend_figure(trends_store)
            meta.update({"input_type": "raw", "fs_hz": fs, "samples": int(len(mv)),
                         "duration_s": round(len(mv) / fs_v, 2)})
            if analysis.get("ok"):
                meta.update({"heart_rate_bpm": analysis["heart_rate"],
                             "pac": f"{analysis['pac_count']} ({analysis['pac_pct']}%)",
                             "pvc": f"{analysis['pvc_count']} ({analysis['pvc_pct']}%)"})
            dur = len(mv) / fs_v
            status = (f"Loaded signal: {filename} — {dur:.1f}s total; "
                      f"showing {start0:.0f}\u2013{min(dur, start0 + window):.0f}s")
        else:
            return (go.Figure(), None, f"Unsupported file type: {filename}",
                    no_update, no_update, no_update, no_update,
                    cleared_report, cleared_store, None, no_update,
                    no_update, no_update, no_update, None)
    except Exception as exc:  # surface parse/render/digitize errors to the user
        return (go.Figure(), None, f"Error: {exc}",
                no_update, no_update, no_update, no_update,
                cleared_report, cleared_store, None, no_update,
                no_update, no_update, no_update, None)

    return (fig, _b64_png(png), status, _b64_png(png), meta,
            _stat_cards(analysis), analysis, cleared_report, cleared_store,
            signal_store, start0, overview_fig, hr_fig, ectopy_fig, trends_store)


@app.callback(
    Output("win-start", "value", allow_duplicate=True),
    Input("win-prev", "n_clicks"),
    Input("win-next", "n_clicks"),
    State("win-start", "value"),
    State("seconds", "value"),
    State("store-signal", "data"),
    prevent_initial_call=True,
)
def nav_window(prev, nxt, start, seconds, sig):
    """Step the display window one page back/forward, clamped to the recording."""
    if not sig:
        return no_update
    window = float(seconds or 10)
    start = float(start or 0)
    dur = float(sig.get("duration") or 0)
    if ctx.triggered_id == "win-next":
        start += window
    elif ctx.triggered_id == "win-prev":
        start -= window
    return round(max(0.0, min(start, max(0.0, dur - window))), 2)


@app.callback(
    Output("ecg-graph", "figure", allow_duplicate=True),
    Output("ecg-image", "src", allow_duplicate=True),
    Output("store-png", "data", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Output("overview-graph", "figure", allow_duplicate=True),
    Input("win-start", "value"),
    Input("seconds", "value"),
    State("store-signal", "data"),
    State("store-analysis", "data"),
    prevent_initial_call=True,
)
def redraw_window(start, seconds, sig, analysis):
    """Re-render only the display window without re-running beat analysis."""
    if not sig:
        return (no_update,) * 5
    mv = np.asarray(sig["mv"], dtype=float)
    fs_v = float(sig["fs"])
    window = float(seconds or 10)
    start = float(start or 0)
    dur = float(sig.get("duration") or len(mv) / fs_v)
    png = sio.render_ecg_png(mv, fs_v, seconds=window, start=start)
    fig = _ecg_figure(mv, fs_v, analysis, "Single-lead ECG — PAC/PVC marked",
                      start=start, seconds=window)
    overview = _overview_figure(mv, fs_v, analysis, start, window)
    status = f"{dur:.1f}s total; showing {start:.0f}\u2013{min(dur, start + window):.0f}s"
    return fig, _b64_png(png), _b64_png(png), status, overview


@app.callback(
    Output("win-start", "value", allow_duplicate=True),
    Input("win-pvc", "n_clicks"),
    Input("win-pac", "n_clicks"),
    State("win-start", "value"),
    State("seconds", "value"),
    State("store-analysis", "data"),
    State("store-signal", "data"),
    prevent_initial_call=True,
)
def jump_event(pvc, pac, start, seconds, analysis, sig):
    """Center the window on the next PVC/PAC after the current position."""
    if not analysis or not sig:
        return no_update
    kind = "PVC" if ctx.triggered_id == "win-pvc" else "PAC"
    t = eca.next_event_time(analysis, float(start or 0), kind)
    if t is None:
        return no_update
    window = float(seconds or 10)
    dur = float(sig.get("duration") or 0)
    return round(max(0.0, min(t - window / 2, max(0.0, dur - window))), 2)


@app.callback(
    Output("win-start", "value", allow_duplicate=True),
    Input("overview-graph", "clickData"),
    State("seconds", "value"),
    State("store-signal", "data"),
    prevent_initial_call=True,
)
def overview_click(click, seconds, sig):
    """Click the overview to move the detail window there."""
    if not click or not sig:
        return no_update
    x = float(click["points"][0]["x"])
    window = float(seconds or 10)
    dur = float(sig.get("duration") or 0)
    return round(max(0.0, min(x - window / 2, max(0.0, dur - window))), 2)


@app.callback(
    Output("win-start", "value", allow_duplicate=True),
    Output("seconds", "value", allow_duplicate=True),
    Input("ecg-graph", "relayoutData"),
    State("store-signal", "data"),
    State("win-start", "value"),
    State("seconds", "value"),
    prevent_initial_call=True,
)
def zoom_to_window(relayout, sig, cur_start, cur_secs):
    """Map a Plotly x zoom/pan to the display window so more/less signal loads."""
    if not sig or not relayout:
        return no_update, no_update
    dur = float(sig.get("duration") or 0)
    # Double-click / reset axes -> back to the first 10 s window.
    if relayout.get("xaxis.autorange") or relayout.get("autosize"):
        if abs(float(cur_start or 0)) < 0.5 and abs(float(cur_secs or 0) - 10) < 0.5:
            return no_update, no_update
        return 0, 10
    if "xaxis.range[0]" in relayout and "xaxis.range[1]" in relayout:
        x0 = max(0.0, float(relayout["xaxis.range[0]"]))
        x1 = float(relayout["xaxis.range[1]"])
        if dur:
            x1 = min(dur, x1)
        win = max(1.0, round(x1 - x0, 2))
        # Ignore no-op relayouts (e.g. echoes of a server-set range) to avoid loops.
        if abs(x0 - float(cur_start or 0)) < 0.5 and abs(win - float(cur_secs or 0)) < 0.5:
            return no_update, no_update
        return round(x0, 2), win
    return no_update, no_update


@app.callback(
    Output("report", "children"),
    Output("store-report", "data"),
    Input("run-btn", "n_clicks"),
    State("store-png", "data"),
    State("prompt", "value"),
    State("store-analysis", "data"),
    prevent_initial_call=True,
)
def run_inference_cb(n_clicks, store_png, prompt, analysis):
    if not store_png:
        return "Upload a signal or image first.", no_update
    report = narrative.generate_report(_facts_from_analysis(analysis),
                                       prompt or DEFAULT_PROMPT)
    combined = eca.format_text(analysis) + "\n\n" + report
    return combined, combined



@app.callback(
    Output("download-pdf", "data"),
    Input("pdf-btn", "n_clicks"),
    State("store-signal", "data"),
    State("store-analysis", "data"),
    State("store-report", "data"),
    State("store-meta", "data"),
    State("store-trends", "data"),
    State("store-png", "data"),
    State("seconds", "value"),
    prevent_initial_call=True,
)
def export_pdf_cb(n_clicks, sig, analysis, report, meta, trends, store_png, seconds):
    if not report:
        return no_update
    if sig and analysis and analysis.get("ok"):
        pdf_bytes = build_full_pdf(
            np.asarray(sig["mv"], dtype=float), float(sig["fs"]), analysis, report,
            meta or {}, window=float(seconds or 10),
            trends=trends if (trends and trends.get("ok")) else None)
    elif store_png:  # image without a recovered signal -> simple single-image PDF
        pdf_bytes = build_pdf(base64.b64decode(store_png.split(",", 1)[1]), report, meta or {})
    else:
        return no_update
    fname = f"pulse_ecg_report_{dt.datetime.now():%Y%m%d_%H%M%S}.pdf"
    return dcc.send_bytes(lambda b: b.write(pdf_bytes), fname)


def _file_to_png(contents, filename, fs, column, unit, seconds):
    """Decode one upload to (png_bytes, input_type, analysis). Raises on bad input."""
    data = _decode_upload(contents)
    if sio.is_image(filename):
        return data, "image", {"ok": False, "reason": "raw signal required"}
    if sio.is_signal(filename):
        sig = sio.parse_signal_bytes(data, filename, column=int(column or 0))
        mv = sio.signal_to_mv(sig, unit=unit)
        fs_v = float(fs or 500)
        png = sio.render_ecg_png(mv, fs_v, seconds=float(seconds or 10))
        analysis = eca.analyze(mv, fs_v)
        _add_rhythm(analysis, mv, fs_v)
        return png, "raw", analysis
    raise ValueError(f"unsupported file type: {filename}")


@app.callback(
    Output("batch-table", "data"),
    Output("store-batch", "data"),
    Output("batch-status", "children"),
    Input("run-batch-btn", "n_clicks"),
    State("batch-upload", "contents"),
    State("batch-upload", "filename"),
    State("batch-prompt", "value"),
    State("fs", "value"),
    State("column", "value"),
    State("unit", "value"),
    State("seconds", "value"),
    prevent_initial_call=True,
)
def run_batch_cb(n_clicks, contents_list, names, prompt, fs, column, unit, seconds):
    if not contents_list:
        return no_update, no_update, "Upload one or more files first."

    table, records = [], []
    for contents, name in zip(contents_list, names):
        try:
            png, itype, analysis = _file_to_png(contents, name, fs, column, unit, seconds)
            model_report = narrative.generate_report(
                _facts_from_analysis(analysis), prompt or DEFAULT_PROMPT)
            report = eca.format_text(analysis) + "\n\n" + model_report
            ok_a = analysis.get("ok")
            records.append({"id": name, "png": _b64_png(png), "report": report,
                            "meta": {"file": name, "input_type": itype}})
            table.append({
                "id": name, "type": itype,
                "HR": analysis["heart_rate"] if ok_a else "-",
                "PAC %": f"{analysis['pac_count']} ({analysis['pac_pct']}%)" if ok_a else "-",
                "PVC %": f"{analysis['pvc_count']} ({analysis['pvc_pct']}%)" if ok_a else "-",
                "report": model_report, "status": "ok",
            })
        except Exception as exc:
            table.append({"id": name, "type": "-", "HR": "-", "PAC %": "-",
                          "PVC %": "-", "report": str(exc), "status": "error"})

    ok = sum(1 for r in table if r["status"] == "ok")
    return table, records, f"Processed {ok}/{len(table)} files."


@app.callback(
    Output("download-batch-jsonl", "data"),
    Input("batch-jsonl-btn", "n_clicks"),
    State("store-batch", "data"),
    prevent_initial_call=True,
)
def export_batch_jsonl_cb(n_clicks, records):
    if not records:
        return no_update
    import json
    lines = "\n".join(json.dumps({"id": r["id"], "report": r["report"],
                                  "meta": r["meta"]}) for r in records)
    fname = f"pulse_batch_{dt.datetime.now():%Y%m%d_%H%M%S}.jsonl"
    return dcc.send_string(lines, fname)


@app.callback(
    Output("download-batch-pdf", "data"),
    Input("batch-pdf-btn", "n_clicks"),
    State("store-batch", "data"),
    prevent_initial_call=True,
)
def export_batch_pdf_cb(n_clicks, records):
    if not records:
        return no_update
    pdf_records = [{
        "id": r["id"],
        "image_bytes": base64.b64decode(r["png"].split(",", 1)[1]),
        "report": r["report"],
        "meta": r["meta"],
    } for r in records]
    pdf_bytes = build_batch_pdf(pdf_records)
    fname = f"pulse_batch_{dt.datetime.now():%Y%m%d_%H%M%S}.pdf"
    return dcc.send_bytes(lambda b: b.write(pdf_bytes), fname)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=False)
