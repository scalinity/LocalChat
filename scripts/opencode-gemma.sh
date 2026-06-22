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
watchdog_pid=""
PIDFILE="$GEMMA_DIR/logs/mlx-server.pid"

# Stop a server we started, escalating TERM -> KILL so it can't linger and
# keep slamming the GPU.
stop_server() {
  local pid="$1"
  kill -0 "$pid" 2>/dev/null || return 0
  echo "Stopping local Gemma server (pid $pid) — freeing RAM…"
  kill -TERM "$pid" 2>/dev/null
  for _ in $(seq 1 20); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.25
  done
  kill -KILL "$pid" 2>/dev/null
}

cleanup() {
  trap - EXIT INT TERM HUP                 # disarm: cleanup runs exactly once
  [[ -n "$watchdog_pid" ]] && kill "$watchdog_pid" 2>/dev/null
  if [[ "$started" == 1 && -n "$srv_pid" ]]; then
    stop_server "$srv_pid"
    wait "$srv_pid" 2>/dev/null
  fi
  rm -f "$PIDFILE"
}
# HUP matters most: closing the terminal sends SIGHUP, whose default action is
# to kill the script *without* running an EXIT-only trap — that is exactly how
# the server got orphaned before. Trapping it routes through cleanup instead.
trap cleanup EXIT INT TERM HUP

# Reap a server orphaned by a previous run that never got to clean up (launcher
# SIGKILLed, machine slept/panicked). The pidfile only ever names a server WE
# started, so killing it can't disturb a shared serve.sh server.
if [[ -f "$PIDFILE" ]]; then
  old_pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null \
     && ps -p "$old_pid" -o command= 2>/dev/null | grep -q "mlx_lm.server"; then
    echo "Reaping orphaned MLX server from a previous session (pid $old_pid)…"
    stop_server "$old_pid"
  fi
  rm -f "$PIDFILE"
fi

# Reuse a server that's already up (e.g. ./serve.sh or another session) and
# leave its lifecycle to whoever started it. Only manage one we start here.
if curl -sf "$BASE/models" >/dev/null 2>&1; then
  echo "Using existing MLX server on $BASE"
else
  echo "Starting local MLX server on $BASE (a model loads on your first message)…"
  "$SERVER" --host 127.0.0.1 --port "$PORT" --log-level INFO >>"$LOG" 2>&1 &
  srv_pid=$!
  started=1
  echo "$srv_pid" > "$PIDFILE"

  # Watchdog: a backstop for cases a trap can't reliably catch. It outlives us
  # and watches TWO lifelines:
  #   1. this launcher    — covers SIGKILL (kill -9) / panic, where no handler runs;
  #   2. the parent shell — covers closing the terminal: bash defers a trapped
  #      signal until the foreground `opencode` exits, so the launcher can be
  #      stuck while the terminal is already gone. The parent dying is the
  #      reliable signal that the session is over.
  # When either lifeline drops, it kills the server AND the (possibly stuck)
  # launcher, so nothing keeps running headless.
  launcher_pid=$$
  parent_pid=$PPID
  (
    while kill -0 "$launcher_pid" 2>/dev/null && kill -0 "$parent_pid" 2>/dev/null; do
      sleep 2
    done
    kill -TERM "$srv_pid" 2>/dev/null
    kill -TERM "$launcher_pid" 2>/dev/null
  ) &
  watchdog_pid=$!
  disown "$watchdog_pid" 2>/dev/null || true

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
