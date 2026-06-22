# Local Chat — desktop app

A small, **offline** macOS chat UI for the local MLX Gemma models in this project.
It is the GUI sibling of `scripts/chat_pretty.py` and reuses the same `.venv` /
`mlx_lm` stack (thinking/answer split, sampler + `stream_generate`, the throttle),
upgraded to real Markdown + KaTeX in a `pywebview` window.

See `docs/DESKTOP_APP_SPEC.md` for the full specification.

## Run (the supported v1 path)

From the **project root**, using the project venv so the patched `mlx_lm`
gemma4 shims load:

```bash
.venv/bin/python desktop/app.py
```

That's it — no build step, no server, no network. The window is frameless: use
the in-app dots (top-left) to close / minimize, and drag the strip between the
model picker and the right-hand buttons to move the window.

## What it does

- **Pick a model** (top-left picker or the start screen) — discovered by scanning
  `models/hub/models--*` (and `*.gguf` files). Loading one unloads the previous
  model and calls `mx.clear_cache()` so RAM returns. The picker has a **Rescan
  models** entry, so models added with the `add` command appear without
  relaunching. Hovering a model row shows a **trash icon** that permanently
  deletes that model's files after a confirm — this is the **one** place the app
  writes under `models/` (a deliberate, user-requested override of the read-only
  rule in §3.2; guarded so it can only touch paths inside `models/`). Adding
  models stays in the terminal (`add` command), so the offline guarantee holds.
- **Chat** multi-turn with a live-streaming answer. The model's hidden *thinking*
  trace streams into a separate, collapsible panel that auto-collapses when the
  answer begins.
- **Render** Markdown, fenced code (highlight.js), and math (`$…$`, `$$…$$`) via
  KaTeX. Plain prose like "$5 to $10" stays literal — it is not parsed as math.
- **Settings** (gear → drawer): temperature, top-p, top-k, min-p, repetition
  penalty + window, max tokens, seed, and a system prompt. Changes **auto-save**
  (no Save button) and apply to the next turn. Scope toggle: **This model** vs
  **All models**. "Reset to defaults" clears the chosen scope.

## Models & formats

Add models with the existing command (downloads into `models/`, then **Rescan**
in the picker):

```bash
./scripts/add-model.sh <hf-repo-id> ["Display name"]
```

Three kinds are supported:

- **MLX models** (`mlx-community/...` quantized): the primary, fastest path —
  run on the Apple GPU via `mlx_lm`.
- **Regular Hugging Face models** (full-precision Llama, Qwen, Mistral, Phi, …):
  also run through `mlx_lm`, which supports ~120 architectures and converts HF
  `safetensors` to MLX at load. No extra setup — drop them in `models/` and pick
  them. Unsupported architectures surface a clear, non-fatal error.
- **GGUF** (llama.cpp `.gguf` files): discovered from `models/gguf/*.gguf` or
  downloaded GGUF repos, run via the `GGUFBackend` (llama-cpp-python).

### GGUF prerequisite — `llama-cpp-python` (+ beta-SDK shim)

GGUF needs `llama-cpp-python` in the project venv. **It's already installed**
(0.3.31, built with Metal). To reinstall/rebuild it, you need the SDK shim in
`packaging/sdk-shim/` (see below):

```bash
SHIM=desktop/packaging/sdk-shim
CMAKE_ARGS="-DGGML_METAL=on -DGGML_ACCELERATE=off -DGGML_BLAS=off \
  -DCMAKE_C_FLAGS=-I$PWD/$SHIM -DCMAKE_CXX_FLAGS=-I$PWD/$SHIM \
  -DCMAKE_OBJC_FLAGS=-I$PWD/$SHIM -DCMAKE_OBJCXX_FLAGS=-I$PWD/$SHIM" \
  .venv/bin/pip install llama-cpp-python
```

> **Why the shim:** this machine runs **macOS 27.0 beta** with the matching beta
> Command Line Tools (no full Xcode). That SDK is missing several legacy
> CoreServices/Carbon headers (`CarbonCore/MacErrors.h`, `SearchKit/SKAnalysis.h`,
> `LangAnalysis/LangAnalysis.h`, `CoreServices/CSIdentity*.h`) that umbrella
> headers like `Foundation.h` still `#include`, so llama.cpp would not compile in
> any configuration. `packaging/sdk-shim/` provides empty stubs for exactly those
> headers (they're only `#include`d, never used by the compiled code) and is put
> on the compiler's include path via the `-I` flags above. Delete the shim once
> Apple ships a complete SDK; it's harmless to keep. This affects **any** native
> Python extension you build on this machine, not just llama.cpp.

Note: thinking traces for GGUF/HF models are detected from textual
`<think>…</think>` tags (the token-level marker split is Gemma-specific), so
non-reasoning models simply stream a plain answer.

## Where things live

| File | Role |
|---|---|
| `app.py` | entry point: builds the frameless window, wires the bridge |
| `api.py` | `js_api` methods exposed to JS; streams tokens back via `evaluate_js` (json-encoded, ~12 Hz) |
| `chatcore.py` | shared backend: `list_models`, `load`, `ThinkingSplitter`, `generate`, `DEFAULT_PARAMS`, `validate_params`, `mathify` |
| `session.py` | in-memory chat state: model handle, prompt cache, transcript, stop flag |
| `settings.py` | sparse runtime settings persistence + `resolve_params` |
| `web/` | `index.html`, `styles.css`, `app.js`, and `vendor/` (markdown-it, KaTeX + fonts, highlight.js, texmath) — all local |

### Inference parameters

Defaults live in exactly one place — `chatcore.DEFAULT_PARAMS`. They are **not**
per-model presets and are not meant to be tuned by editing source: change them at
runtime in the Settings panel. User overrides persist **sparsely** (only the keys
you changed) to:

```
~/Library/Application Support/LocalChat/settings.json
```

That file does not exist until your first change. A model you never customize
always resolves to pure `DEFAULT_PARAMS` (plus any "All models" overlay).

## Settings file

The only file the app ever writes is `settings.json` above. Nothing under
`models/` is touched (model files are read-only), and conversations live in
memory only (no chat history on disk in v1).

## Dev checks (optional)

- `.venv/bin/python desktop/_smoke.py` — headless backend proof against `e4b`
  (load, thinking/answer split, no marker leakage, RAM release).
- `.venv/bin/python desktop/_app_test.py` — drives the real window end-to-end
  (select model → stream a reply) and verifies the bridge + rendering.

## Packaging

Building a standalone `.app` is a **stretch goal, not part of v1** — see
`docs/DESKTOP_APP_SPEC.md` §14 for why (Python 3.14 + a monkeypatched `mlx_lm`
in site-packages is the riskiest path to bundle). For now, run from the venv as
shown above. `packaging/` is a placeholder for that future work.
