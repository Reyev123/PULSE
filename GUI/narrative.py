"""Generate the ECG narrative from measured facts using a local Ollama LLM.

Replaces the PULSE VLM for report text: the signal analyzer / RhythmCNN provide
the facts, a small local model writes the prose. No GPU model load required.
"""

import json
import os
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

_SYSTEM = (
    "You are a cardiology assistant. Write a concise clinical narrative for a "
    "SINGLE-LEAD ECG using ONLY the automated measurements provided. Do not "
    "invent 12-lead findings (cardiac axis, lead-specific localization, "
    "R-wave progression). Keep it factual and short. End with: "
    "'Research use only - not a diagnosis.'"
)


def available():
    """True if the Ollama server responds and the model is present."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            tags = json.loads(r.read())
        names = [m.get("name", "") for m in tags.get("models", [])]
        return any(n == OLLAMA_MODEL or n.startswith(OLLAMA_MODEL) for n in names)
    except Exception:
        return False


def _facts_text(facts):
    parts = []
    if facts.get("heart_rate") is not None:
        parts.append(f"heart rate {facts['heart_rate']} bpm")
    if facts.get("pac") is not None:
        parts.append(f"PACs {facts['pac']}")
    if facts.get("pvc") is not None:
        parts.append(f"PVCs {facts['pvc']}")
    if facts.get("rhythm"):
        parts.append(f"rhythm classifier: {facts['rhythm']}")
    if facts.get("total_beats") is not None:
        parts.append(f"total beats {facts['total_beats']}")
    return "; ".join(parts) if parts else "no reliable measurements available"


def generate_report(facts, user_prompt, model=OLLAMA_MODEL):
    """Return report text, or an error string if Ollama is unavailable."""
    if not available():
        return (f"[Ollama model '{model}' not available at {OLLAMA_URL}. "
                "Start Ollama and pull the model to enable narrative generation.]")
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content":
                f"Automated measurements: {_facts_text(facts)}.\n"
                f"Task: {user_prompt}"},
        ],
        "stream": False,
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["message"]["content"].strip()
    except Exception as exc:
        return f"[Narrative generation error: {exc}]"
