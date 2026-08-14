#!/usr/bin/env bash
# Start the local (Spark) Ollama that the AWS deployment can use as its LLM
# backend over Tailscale. Serves on all interfaces so the tailnet can reach it,
# ensures the model is present, and prints the tailnet URL to use as
# PULSE_OLLAMA_URL / PULSE_OLLAMA_FALLBACK_URL.
set -euo pipefail

MODEL="${PULSE_OLLAMA_MODEL:-llama3.1:8b}"
PORT="${OLLAMA_PORT:-11434}"
LOG="${OLLAMA_LOG:-/tmp/pulse_ollama.log}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "ollama is not installed. See https://ollama.com/download" >&2
  exit 1
fi

if curl -fsS "http://127.0.0.1:${PORT}/api/tags" >/dev/null 2>&1; then
  echo "Ollama already running on port ${PORT}."
else
  echo "Starting ollama serve on 0.0.0.0:${PORT} (log: ${LOG})..."
  OLLAMA_HOST="0.0.0.0:${PORT}" nohup ollama serve >"${LOG}" 2>&1 &
  for _ in $(seq 1 30); do
    curl -fsS "http://127.0.0.1:${PORT}/api/tags" >/dev/null 2>&1 && break
    sleep 1
  done
fi

if ! curl -fsS "http://127.0.0.1:${PORT}/api/tags" | grep -q "${MODEL%%:*}"; then
  echo "Pulling ${MODEL}..."
  OLLAMA_HOST="127.0.0.1:${PORT}" ollama pull "${MODEL}"
fi

echo "Local Ollama ready with ${MODEL}."
if command -v tailscale >/dev/null 2>&1; then
  TS_IP="$(tailscale ip -4 2>/dev/null | head -n1 || true)"
  if [[ -n "${TS_IP}" ]]; then
    echo "Tailnet URL (use for PULSE_OLLAMA_URL / PULSE_OLLAMA_FALLBACK_URL):"
    echo "  http://${TS_IP}:${PORT}"
  else
    echo "Tailscale installed but no tailnet IP; run: tailscale up"
  fi
else
  echo "Tailscale not found; install it so AWS can reach this endpoint."
fi
