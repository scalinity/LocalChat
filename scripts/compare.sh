#!/usr/bin/env bash
# compare.sh — give the same Three.js build task to each Gemma 4 model,
# save runnable HTML + timing, so you can compare side-by-side.
set -euo pipefail
cd "$(dirname "$0")/.."   # scripts/ lives under the project root

export HF_HOME="$PWD/models"
GEN="$PWD/.venv/bin/mlx_lm.generate"
PY="$PWD/.venv/bin/python"
OUT="compare_out"
mkdir -p "$OUT"

# Tunables
MAX_TOKENS="${MAX_TOKENS:-8000}"
TEMP="${TEMP:-0.2}"

# repo-id<TAB>short-name
MODELS=(
  "mlx-community/gemma-4-12B-it-qat-6bit	12b-6bit"
  "mlx-community/gemma-4-31B-it-qat-4bit	31b-4bit"
  "mlx-community/gemma-4-26b-a4b-8bit	26b-a4b-8bit"
)

# The shared coding task. Single self-contained file = easy to open & judge.
read -r -d '' PROMPT <<'EOF' || true
Create a SINGLE self-contained HTML file using Three.js that renders an
interactive 3D solar system. Requirements:
- Load Three.js r160+ and OrbitControls via an ES module importmap from a CDN
  (esm.sh or unpkg). No build step, no npm.
- A glowing sun at the center plus at least 6 planets orbiting at different
  speeds and distances, each rotating on its own axis.
- Realistic-ish: point light at the sun, planets with distinct colors/sizes,
  faint orbit rings, and a starfield background of a few thousand points.
- OrbitControls so the user can drag to rotate and scroll to zoom.
- Responsive: resize with the window. Animation via requestAnimationFrame.
Output ONLY the complete HTML in one ```html code block — no explanation.
EOF

echo "Task: Three.js interactive solar system"
echo "Settings: max_tokens=$MAX_TOKENS temp=$TEMP"
echo "Output dir: $OUT/"
echo

SUMMARY="$OUT/SUMMARY.md"
{ echo "# Gemma 4 Three.js build comparison"; echo; echo "| model | gen tokens | tokens/sec | wall sec | html bytes |"; echo "|---|---|---|---|---|"; } > "$SUMMARY"

for entry in "${MODELS[@]}"; do
  repo="${entry%%	*}"
  name="${entry##*	}"
  raw="$OUT/$name.raw.txt"
  html="$OUT/$name.html"

  echo "=== $name ($repo) ==="
  if [ ! -d models/hub/models--${repo//\//--} ]; then
    echo "  SKIP — not downloaded yet"; echo
    continue
  fi

  start=$(date +%s)
  "$GEN" --model "$repo" --prompt "$PROMPT" \
         --max-tokens "$MAX_TOKENS" --temp "$TEMP" \
         > "$raw" 2>&1 || { echo "  generation failed, see $raw"; echo; continue; }
  end=$(date +%s); wall=$((end-start))

  # Extract the largest ```...``` code block into a runnable .html
  "$PY" - "$raw" "$html" <<'PYEOF'
import re, sys
raw, out = sys.argv[1], sys.argv[2]
txt = open(raw, encoding="utf-8", errors="replace").read()
blocks = re.findall(r"```[a-zA-Z]*\n(.*?)```", txt, re.S)
code = max(blocks, key=len) if blocks else txt
open(out, "w", encoding="utf-8").write(code.strip()+"\n")
print(f"  wrote {out} ({len(code)} chars)")
PYEOF

  # Pull mlx's stats line: "Generation: N tokens, ... X tokens-per-sec"
  toks=$(grep -oE 'Generation:[^,]*' "$raw" | grep -oE '[0-9]+' | head -1 || echo "?")
  tps=$(grep -oE '[0-9.]+ tokens-per-sec' "$raw" | tail -1 | grep -oE '[0-9.]+' || echo "?")
  bytes=$(wc -c < "$html" | tr -d ' ')
  echo "  tokens=$toks  tok/s=$tps  wall=${wall}s  html=${bytes}B"
  echo "| $name | $toks | $tps | $wall | $bytes |" >> "$SUMMARY"
  echo
done

echo "Done. Summary: $SUMMARY"
echo "Open a result in your browser, e.g.:"
echo "  open $OUT/31b-4bit.html"
