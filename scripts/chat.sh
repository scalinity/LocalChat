#!/usr/bin/env bash
# chat.sh — open an interactive terminal chat with ONE local Gemma 4 model.
# Mirrors serve.sh but runs mlx_lm's built-in REPL instead of a server.
# Usable from anywhere as `chat 12b` (see the `chat` function in ~/.zshrc).
#   chat 12b      # gemma-4-12B-it-qat-6bit
#   chat 31b      # gemma-4-31B-it-qat-4bit
#   chat 26b      # gemma-4-26b-a4b-8bit
# Extra mlx_lm.chat flags pass through, e.g.:
#   chat 12b --temp 0.7 --system-prompt "You are concise."
# In the chat: 'q' quits · 'r' resets the conversation · 'h' shows help.
set -euo pipefail
cd "$(dirname "$0")/.."   # scripts/ lives under the project root
export HF_HOME="$PWD/models"
export HF_HUB_OFFLINE=1   # all 3 models are already downloaded; skip the network

case "${1:-}" in
  12b) REPO="mlx-community/gemma-4-12B-it-qat-6bit" ;;
  31b) REPO="mlx-community/gemma-4-31B-it-qat-4bit" ;;
  26b) REPO="mlx-community/gemma-4-26b-a4b-8bit" ;;
  -h|--help|"") echo "usage: chat [12b|31b|26b|<hf-repo-id>] [extra mlx_lm.chat flags]"; exit 0 ;;
  */*) REPO="$1" ;;   # a full HF repo id, e.g. mlx-community/Qwen3-30B-A3B-4bit
  *) echo "unknown model '$1' — use 12b|31b|26b or a full repo id (mlx-community/...)"; exit 1 ;;
esac

# Pretty REPL: dim thinking trace, clearly separated from the Markdown-rendered
# answer (see chat_pretty.py). A user-supplied --max-tokens/-m overrides 4096.
exec .venv/bin/python scripts/chat_pretty.py --model "$REPO" --max-tokens 4096 "${@:2}"
