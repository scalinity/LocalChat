#!/usr/bin/env bash
# add-model.sh — download an MLX model into the LOCAL cache (models/) and
# register it in opencode's GLOBAL config so it appears in the model picker
# everywhere. Afterwards it also works with `chat <repo>` and `./serve.sh`.
#
#   ./add-model.sh mlx-community/Qwen3-30B-A3B-4bit
#   ./add-model.sh mlx-community/Qwen3-30B-A3B-4bit "Qwen3 30B A3B (4bit)"
#
# Tip: pick MLX builds (the "mlx-community/..." repos) so they run on Apple GPU.
set -euo pipefail
cd "$(dirname "$0")/.."                # scripts/ lives under the project root
export HF_HOME="$PWD/models"          # keep everything in this project's cache

REPO="${1:-}"
if [[ -z "$REPO" ]]; then
  echo "usage: ./add-model.sh <hf-repo-id> [display-name]"; exit 1
fi
NAME="${2:-$REPO}"
GLOBAL_CFG="$HOME/.config/opencode/opencode.jsonc"

echo "==> Downloading $REPO into $HF_HOME (may be large)…"
.venv/bin/hf download "$REPO" >/dev/null

echo "==> Checking mlx-lm supports this architecture…"
.venv/bin/python - "$REPO" <<'PY'
import os, json, glob, importlib, sys
repo = sys.argv[1]
hub = os.path.join(os.environ["HF_HOME"], "hub", "models--" + repo.replace("/", "--"))
cfgs = glob.glob(os.path.join(hub, "snapshots", "*", "config.json"))
if not cfgs:
    print("  ! no config.json found; skipping check"); sys.exit(0)
mt = json.load(open(cfgs[0])).get("model_type", "?")
try:
    from mlx_lm.utils import MODEL_REMAPPING
    importlib.import_module(f"mlx_lm.models.{MODEL_REMAPPING.get(mt, mt)}")
    print(f"  ok — model_type '{mt}' is supported")
except Exception as e:
    print(f"  WARNING — model_type '{mt}' is not natively supported: {e}")
    print("  It may need a newer mlx-lm or a shim (see models/gemma4_unified.py example).")
PY

echo "==> Registering in opencode config: $GLOBAL_CFG"
mkdir -p "$(dirname "$GLOBAL_CFG")"
[[ -f "$GLOBAL_CFG" ]] || echo '{ "$schema": "https://opencode.ai/config.json" }' > "$GLOBAL_CFG"
.venv/bin/python - "$GLOBAL_CFG" "$REPO" "$NAME" <<'PY'
import json, sys
cfg_path, repo, name = sys.argv[1], sys.argv[2], sys.argv[3]
with open(cfg_path) as f:
    cfg = json.load(f)
mlx = cfg.setdefault("provider", {}).setdefault("mlx", {})
mlx.setdefault("npm", "@ai-sdk/openai-compatible")
mlx.setdefault("name", "MLX (local)")
mlx.setdefault("options", {"baseURL": "http://127.0.0.1:8080/v1", "apiKey": "local"})
mlx.setdefault("models", {})[repo] = {"name": name}
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
print(f'  added: {repo}  ->  "{name}"')
PY

echo
echo "Done. Use it from anywhere:"
echo "  - opencode : /models -> MLX (local) -> $NAME"
echo "  - terminal : chat $REPO"
