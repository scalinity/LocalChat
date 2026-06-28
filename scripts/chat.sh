#!/usr/bin/env bash
# chat.sh — open an interactive terminal chat with ONE local Gemma 4 model.
# Runs mlx_lm's built-in REPL; the model loads on launch and is freed on quit.
# Usable from anywhere as `chat 12b` (see the `chat` function in ~/.zshrc).
#   chat 12b      # gemma-4-12B-it-qat-6bit
#   chat 31b      # gemma-4-31B-it-qat-4bit
#   chat 26b      # gemma-4-26b-a4b-8bit
#   chat 4b       # gemma-4-e4b-it-4bit
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
  26b) REPO="Jiunsong/supergemma4-26b-mlx-4bit-v2" ;;
  4b|e4b) REPO="mlx-community/gemma-4-E4B-it-qat-4bit" ;;
  -h|--help|"") echo "usage: chat [12b|31b|26b|4b|<hf-repo-id>|<name.gguf>] [extra mlx_lm.chat flags]"; exit 0 ;;
  *.gguf)   # a GGUF file (llama.cpp via chatcore): full path or a bare filename
    if [[ -f "$1" ]]; then
      REPO="$1"
    else                                  # resolve a bare name from the local cache
      # The scan is bounded by directory entries (models/hub is large on disk but
      # shallow), so this stays sub-second. Refuse an ambiguous bare name rather
      # than silently picking whichever copy `find` happens to list first.
      matches=$(find "$PWD/models/gguf" "$PWD/models/hub" -name "$(basename "$1")" 2>/dev/null)
      n=$(printf '%s\n' "$matches" | grep -c .)
      if [[ "$n" -eq 0 ]]; then
        echo "gguf model '$1' not found under models/"; exit 1
      elif [[ "$n" -gt 1 ]]; then
        echo "gguf model '$1' is ambiguous — $n matches; pass the full path:"; echo "$matches"; exit 1
      fi
      REPO="$matches"
    fi ;;
  */*) REPO="$1" ;;   # a full HF repo id, e.g. mlx-community/Qwen3-30B-A3B-4bit
  *) echo "unknown model '$1' — use 12b|31b|26b|4b, a repo id (mlx-community/...) or a name.gguf"; exit 1 ;;
esac

# Pretty REPL: dim thinking trace, clearly separated from the Markdown-rendered
# answer (see chat_pretty.py). Generation runs unbounded until the model stops.
exec .venv/bin/python scripts/chat_pretty.py --model "$REPO" "${@:2}"
