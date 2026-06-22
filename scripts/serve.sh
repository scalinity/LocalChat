#!/usr/bin/env bash
# serve.sh — run a local OpenAI-compatible server for opencode to talk to.
# The server hot-swaps to whichever model a request names, so you only run
# this ONCE and switch models freely inside opencode — no per-model restart.
#
#   ./serve.sh            # serve all 3 (load-on-demand, nothing preloaded)
#   ./serve.sh 12b        # same, but preload gemma-4-12B-it-qat-6bit
#   ./serve.sh 31b        # same, but preload gemma-4-31B-it-qat-4bit
#   ./serve.sh 26b        # same, but preload gemma-4-26b-a4b-8bit
set -euo pipefail
cd "$(dirname "$0")/.."   # scripts/ lives under the project root
export HF_HOME="$PWD/models"

case "${1:-}" in
  ""|all)
    echo "Serving all Gemma 4 models on http://127.0.0.1:8080/v1"
    echo "(load-on-demand — pick any model in opencode; Ctrl-C to stop)"
    exec .venv/bin/mlx_lm.server --port 8080
    ;;
  12b) REPO="mlx-community/gemma-4-12B-it-qat-6bit" ;;
  31b) REPO="mlx-community/gemma-4-31B-it-qat-4bit" ;;
  26b) REPO="mlx-community/gemma-4-26b-a4b-8bit" ;;
  -h|--help) echo "usage: ./serve.sh [all|12b|31b|26b|<hf-repo-id>]"; exit 0 ;;
  */*) REPO="$1" ;;   # preload a full HF repo id; others still load on demand
  *) echo "usage: ./serve.sh [all|12b|31b|26b|<hf-repo-id>]"; exit 1 ;;
esac

echo "Serving $REPO on http://127.0.0.1:8080/v1"
echo "(preloaded; other models still load on demand; Ctrl-C to stop)"
exec .venv/bin/mlx_lm.server --model "$REPO" --port 8080
