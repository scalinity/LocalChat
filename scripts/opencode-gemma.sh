#!/usr/bin/env bash
# opencode-gemma.sh — run opencode (in your CURRENT directory) backed by a
# LOCAL Gemma 4 MLX server that lives only for this session: started on launch,
# stopped — RAM freed — when opencode exits.
#
# Works from ANY directory: the server uses the models + venv under
# ~/Documents/Gemma, while opencode itself runs wherever you launched it, so it
# operates on your current project. The MLX provider/models are registered in
# opencode's GLOBAL config (~/.config/opencode/opencode.jsonc), so they're
# selectable everywhere.
set -uo pipefail

GEMMA_DIR="$HOME/Documents/Gemma"     # local models + venv live here
export HF_HOME="$GEMMA_DIR/models"
export HF_HUB_OFFLINE=1               # all 3 models are downloaded; skip network

PORT=8080
BASE="http://127.0.0.1:$PORT/v1"
LOG="$GEMMA_DIR/logs/mlx-server.log"
SERVER="$GEMMA_DIR/.venv/bin/mlx_lm.server"

started=0
srv_pid=""

cleanup() {
  if [[ "$started" == 1 && -n "$srv_pid" ]] && kill -0 "$srv_pid" 2>/dev/null; then
    echo "Stopping local Gemma server (pid $srv_pid) — freeing RAM…"
    kill "$srv_pid" 2>/dev/null
    wait "$srv_pid" 2>/dev/null
  fi
}
trap cleanup EXIT INT TERM

# Reuse a server that's already up (e.g. ./serve.sh or another session) and
# leave its lifecycle to whoever started it. Only manage one we start here.
if curl -sf "$BASE/models" >/dev/null 2>&1; then
  echo "Using existing MLX server on $BASE"
else
  echo "Starting local MLX server on $BASE (a model loads on your first message)…"
  "$SERVER" --host 127.0.0.1 --port "$PORT" --log-level INFO >>"$LOG" 2>&1 &
  srv_pid=$!
  started=1
  for _ in $(seq 1 40); do
    curl -sf "$BASE/models" >/dev/null 2>&1 && break
    if ! kill -0 "$srv_pid" 2>/dev/null; then
      echo "Server failed to start — see $LOG" >&2
      exit 1
    fi
    sleep 0.5
  done
fi

# Run opencode in the foreground, in the CURRENT directory; when you quit it,
# the trap stops the server we started.
if ! command -v opencode >/dev/null 2>&1; then
  echo "opencode is not installed / not on PATH." >&2
  echo "Install it, then re-run." >&2
  exit 127
fi
command opencode "$@"
