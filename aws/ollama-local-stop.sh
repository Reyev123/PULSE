#!/usr/bin/env bash
# Stop the local (Spark) Ollama started by ollama-local-start.sh. Leaves
# Tailscale running (other services may use it).
set -euo pipefail

PORT="${OLLAMA_PORT:-11434}"

if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet ollama 2>/dev/null; then
  echo "Stopping systemd ollama service..."
  sudo systemctl stop ollama
  exit 0
fi

pids="$(pgrep -f 'ollama serve' || true)"
if [[ -z "${pids}" ]]; then
  echo "No 'ollama serve' process found."
  exit 0
fi

echo "Stopping ollama serve (pids: ${pids})..."
kill ${pids} 2>/dev/null || true
sleep 2
pgrep -f 'ollama serve' >/dev/null 2>&1 && kill -9 $(pgrep -f 'ollama serve') 2>/dev/null || true
echo "Local Ollama on port ${PORT} stopped."
