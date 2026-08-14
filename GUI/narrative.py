"""Generate the ECG narrative from measured facts using a local Ollama LLM.

Replaces the PULSE VLM for report text: the signal analyzer / RhythmCNN provide
the facts, a small local model writes the prose. No GPU model load required.
"""

import ipaddress
import json
import os
import urllib.parse
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
# Optional fallback endpoint(s) tried when OLLAMA_URL is down (e.g. the AWS VM
# is stopped): comma-separated, first reachable wins. Typically the local Spark
# machine's Ollama exposed to the deployment.
OLLAMA_FALLBACK_URL = os.environ.get("OLLAMA_FALLBACK_URL", "")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_OPTIONAL = os.environ.get("OLLAMA_OPTIONAL", "true").lower() == "true"
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "3"))
# host:port of a SOCKS5 proxy (the Tailscale sidecar) to reach tailnet Ollama.
OLLAMA_SOCKS5 = os.environ.get("OLLAMA_SOCKS5", "").strip()
_TAILNET = ipaddress.ip_network("100.64.0.0/10")  # Tailscale CGNAT range


def _via_proxy(url):
    """Proxy only tailnet hosts; VPC/localhost endpoints stay direct."""
    if not OLLAMA_SOCKS5:
        return False
    host = urllib.parse.urlparse(url).hostname or ""
    try:
        return ipaddress.ip_address(host) in _TAILNET
    except ValueError:
        return True  # a hostname (e.g. MagicDNS) is assumed to be on the tailnet


def _opener(url):
    """urllib opener via the SOCKS5 proxy for tailnet URLs, else direct."""
    if not _via_proxy(url):
        return urllib.request.build_opener()
    host, _, port = OLLAMA_SOCKS5.partition(":")
    import socks  # PySocks
    from sockshandler import SocksiPyHandler
    return urllib.request.build_opener(
        SocksiPyHandler(socks.SOCKS5, host, int(port or 1080), rdns=True))


def _endpoints():
    """Ordered, de-duplicated Ollama URLs: primary first, then fallbacks."""
    urls = [OLLAMA_URL] + [u.strip() for u in OLLAMA_FALLBACK_URL.split(",")]
    seen, ordered = set(), []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered

_SYSTEM = (
    "You are a cardiology assistant. Write a concise clinical narrative for a "
    "SINGLE-LEAD ECG using ONLY the automated measurements provided. Write in "
    "flowing prose sentences as one or two short paragraphs; do NOT use bullet "
    "points, lists, or headings. Naturally weave in every measurement provided "
    "(heart rate, total beats, PACs with count and percentage, PVCs with count "
    "and percentage, rhythm) without omitting any. Do not add measurements that "
    "were not given, and do not invent 12-lead findings (cardiac axis, "
    "lead-specific localization, R-wave progression). If signal quality is "
    "'poor', add one sentence cautioning that motion/artifact makes the "
    "findings unreliable. Keep it factual and short. "
    "End with: 'Research use only - not a diagnosis.'"
)


def _endpoint_has_model(url, model=OLLAMA_MODEL):
    """True if this Ollama URL responds and serves the requested model."""
    try:
        with _opener(url).open(f"{url}/api/tags", timeout=OLLAMA_TIMEOUT) as r:
            tags = json.loads(r.read())
        names = [m.get("name", "") for m in tags.get("models", [])]
        return any(n == model or n.startswith(model) for n in names)
    except Exception:
        return False


def active_endpoint(model=OLLAMA_MODEL):
    """First endpoint (primary, then fallbacks) that serves the model, else None."""
    for url in _endpoints():
        if _endpoint_has_model(url, model):
            return url
    return None


def available():
    """True if any configured Ollama endpoint serves the model."""
    return active_endpoint() is not None


def _facts_text(facts):
    parts = []
    if facts.get("heart_rate") is not None:
        parts.append(f"- Heart rate: {facts['heart_rate']} bpm")
    if facts.get("total_beats") is not None:
        parts.append(f"- Total beats: {facts['total_beats']}")
    if facts.get("pac") is not None:
        parts.append(f"- PACs: {facts['pac']}")
    if facts.get("pvc") is not None:
        parts.append(f"- PVCs: {facts['pvc']}")
    if facts.get("rhythm"):
        parts.append(f"- Rhythm classifier: {facts['rhythm']}")
    if facts.get("signal_quality"):
        parts.append(f"- Signal quality: {facts['signal_quality']}")
    return "\n".join(parts) if parts else "- no reliable measurements available"


def generate_report(facts, user_prompt, model=OLLAMA_MODEL):
    """Return report text, or an error string if Ollama is unavailable."""
    endpoint = active_endpoint(model)
    if endpoint is None:
        if OLLAMA_OPTIONAL:
            return ("[Narrative unavailable: no Ollama service is reachable. "
                    "Deterministic ECG measurements remain available above. "
                    "Research use only - not a diagnosis.]")
        return (f"[Ollama model '{model}' not available at {', '.join(_endpoints())}. "
                "Start Ollama and pull the model to enable narrative generation.]")
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content":
                f"Automated measurements (state ALL of these, verbatim, in your "
                f"report):\n{_facts_text(facts)}\n\nTask: {user_prompt}"},
        ],
        "stream": False,
    }
    req = urllib.request.Request(
        f"{endpoint}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with _opener(endpoint).open(req, timeout=120) as r:
            return json.loads(r.read())["message"]["content"].strip()
    except Exception as exc:
        return f"[Narrative generation error: {exc}]"
