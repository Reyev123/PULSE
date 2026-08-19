"""Regression tests for the PULSE GUI callbacks.

Runs without a browser by calling the Dash callback functions directly. Dash's
``@app.callback`` returns the original function, so ``app.nav_window`` etc. are
plain callables; callbacks that read ``dash.ctx`` are exercised by swapping in a
fake context (``app.ctx``).

Run:
    python tests/test_regression.py          # plain, no pytest needed
    pytest tests/test_regression.py -q       # if pytest is installed
"""

import base64
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_GUI = os.path.dirname(_HERE)
if _GUI not in sys.path:
    sys.path.insert(0, _GUI)

import app  # noqa: E402
import ecg_analysis as eca  # noqa: E402

_DEMO = os.path.join(_GUI, "samples", "demo_pac_pvc.csv")

# handle_upload return positions (see app.handle_upload).
I_STATUS, I_ANALYSIS, I_SIGNAL = 2, 6, 9

_cache = {}


def _load_demo(fs=500, unit="mv", seconds=10):
    """Run handle_upload on the demo CSV; return (analysis, signal_store)."""
    key = (fs, unit, seconds)
    if key not in _cache:
        data = open(_DEMO, "rb").read()
        contents = "data:text/csv;base64," + base64.b64encode(data).decode()
        out = app.handle_upload(contents, "demo_pac_pvc.csv", fs, 0, unit, seconds)
        _cache[key] = (out[I_ANALYSIS], out[I_SIGNAL], out)
    return _cache[key][0], _cache[key][1]


def _set_ctx(trigger):
    app.ctx = types.SimpleNamespace(triggered_id=trigger)


def _xrange(fig):
    return [round(float(v), 2) for v in fig.layout.xaxis.range]


# --- upload -----------------------------------------------------------------
def test_upload_parses_and_analyzes():
    a, sig = _load_demo()
    assert a and a.get("ok"), "analysis should succeed on the demo"
    assert a["total_beats"] > 10
    assert sig and sig["duration"] > 20
    assert app.handle_upload(None, None, 500, 0, "mv", 10)[0] is not None  # guarded


# --- navigation moves the detail graph --------------------------------------
def test_next_advances_window():
    a, sig = _load_demo()
    _set_ctx("win-next")
    start = app.nav_window(0, 1, 0, 10, sig)
    assert start == 10, f"Next should page to 10 s, got {start}"
    fig, _status, _ov = app.redraw_window(start, 10, sig, a)
    assert _xrange(fig)[0] == 10, "detail graph x-axis must start at 10 s"


def test_next_clamps_at_end():
    a, sig = _load_demo()
    _set_ctx("win-next")
    # From 10 s, next page (20) exceeds duration-window (14) -> clamps to 14.
    start = app.nav_window(0, 1, 10, 10, sig)
    assert start == round(sig["duration"] - 10, 2), f"should clamp, got {start}"


def test_prev_goes_back():
    a, sig = _load_demo()
    _set_ctx("win-prev")
    start = app.nav_window(1, 0, 14, 10, sig)
    assert start == 4, f"Prev from 14 should be 4, got {start}"
    fig, *_ = app.redraw_window(start, 10, sig, a)
    assert _xrange(fig)[0] == 4


def test_next_pvc_centers_on_a_pvc():
    a, sig = _load_demo()
    _set_ctx("win-pvc")
    start = app.jump_event(1, 0, 0, 10, a, sig)
    assert start is not None and start > 0, "Next PVC should jump forward"
    fig, *_ = app.redraw_window(start, 10, sig, a)
    assert abs(_xrange(fig)[0] - start) < 0.6, "graph window must follow the jump"


def test_next_pac_moves():
    a, sig = _load_demo()
    _set_ctx("win-pac")
    start = app.jump_event(0, 1, 0, 10, a, sig)
    assert start is not None, "Next PAC should return a position"


def test_redraw_is_lightweight():
    # Regression: navigation must not depend on / update the heavy PNG stores.
    a, sig = _load_demo()
    out = app.redraw_window(4, 10, sig, a)
    assert len(out) == 3, "redraw_window should only update graph/status/overview"


# --- export -----------------------------------------------------------------
def test_export_pdf_without_inference():
    a, sig = _load_demo()
    trends = eca.compute_trends(a, duration_s=sig["duration"])
    res, status = app.export_pdf_cb(1, sig, a, None, {"file": "demo"}, trends, None, 10)
    assert isinstance(res, dict), "export should return a download, not no_update"
    raw = base64.b64decode(res["content"])
    assert raw[:4] == b"%PDF", "exported bytes must be a valid PDF"
    assert "PDF ready" in status, "export must report a ready status for the spinner"


def test_export_pdf_no_data_reports_status():
    # Nothing loaded -> no download but a user-facing status (no silent no-op).
    res, status = app.export_pdf_cb(1, None, None, None, {}, None, None, 10)
    assert res is app.no_update
    assert status and "Load" in status


def test_pdf_findings_include_hrv_numbers():
    a, _sig = _load_demo()
    txt = eca.format_text(a)
    assert "Rhythm (HRV second opinion)" in txt
    assert "RMSSD" in txt and "pNN50" in txt and "RR CV" in txt


# --- wiring (catches broken Input/Output registration) ----------------------
def test_callback_wiring():
    keys = list(app.app.callback_map.keys())
    joined = " ".join(keys)
    assert "ecg-graph.figure" in joined, "ecg-graph.figure must be a callback output"
    assert "win-start.value" in joined, "win-start.value must be a callback output"


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        _set_ctx(None)  # reset context between tests
        try:
            fn()
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
        else:
            print(f"ok   {fn.__name__}")
            passed += 1
    print(f"\n{passed}/{len(fns)} passed")
    return passed == len(fns)


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
